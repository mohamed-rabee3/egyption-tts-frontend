"""Client and test for the tashkeel-egy Vast serverless endpoint (plain HTTPS, stdlib only).

    python vast_ai_serverless_test.py            # VAST_API_KEY from the project .env
    python vast_ai_serverless_test.py --text "إزيك عامل إيه"

Use the client from your own code:

    from vast_ai_serverless_test import TashkeelVastClient
    client = TashkeelVastClient()                   # fetches (or reuses) the endpoint token
    client.diacritize("إزيك عامل إيه")["text"]      # -> "إِزَّيَّك عامل إيه"
    client.diacritize_batch(["صباح الخير", "تصبح على خير"])

Every call goes through Vast's three steps:

  1. Token: GET https://console.vast.ai/api/v0/endptjobs/?client_id=me (account key) gives
     the endpoint's own key. It expires (a fixed date, about a week out), so the client keeps
     it in a temp file, reuses it while valid, and fetches a new one when Vast rejects it.
  2. Worker: POST https://run.vast.ai/route/ gives a worker URL and a one-time signed
     auth_data. Done for every request: workers change and signatures are single use.
  3. Request: POST <worker url>/diacritize with {"auth_data": ..., "payload": <API body>}.
     The worker's TLS certificate is signed by Vast's CA (jvastai_root.cer).
"""

import argparse
import base64
import json
import os
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request

CONSOLE = "https://console.vast.ai"
ROUTER = "https://run.vast.ai"
VAST_CA_URL = f"{CONSOLE}/static/jvastai_root.cer"


def load_dotenv(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    """Read KEY=VALUE lines from the project's .env (git-ignored) into os.environ.
    Variables already set in the environment win."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip("'\"")
            os.environ.setdefault(key, value)


class TokenRejected(Exception):
    """Vast refused the endpoint token: fetch a new one and retry."""


class TashkeelVastClient:
    ENDPOINT = "tashkeel-egy"

    def __init__(self, account_key=None, token_file=None, route_wait=180.0, timeout=120.0):
        load_dotenv()
        self.account_key = account_key or os.environ.get("VAST_API_KEY")
        if not self.account_key:
            raise RuntimeError("VAST_API_KEY not found: add it to this project's .env or the environment")
        self.token_file = token_file or os.path.join(
            tempfile.gettempdir(), f"vast_endpoint_token_{self.ENDPOINT}.json")
        self.route_wait = route_wait
        self.timeout = timeout
        self.ssl_context = self._vast_ssl_context()
        self.token = None
        self.fetch_token()

    # ------------------------------------------------------------ step 1: token

    def fetch_token(self, force=False):
        """Return the endpoint token: the saved one while it is valid, else a new one.

        force=True skips the saved token (used after Vast rejected it).
        """
        if not force:
            saved = self._read_saved_token()
            if saved:
                self.token = saved
                return self.token
        status, body = self._http_json(
            "GET", f"{CONSOLE}/api/v0/endptjobs/?client_id=me",
            headers={"Authorization": f"Bearer {self.account_key}"})
        if status != 200:
            raise RuntimeError(f"fetching the endpoint token failed: HTTP {status}: {body}")
        for e in body["results"]:
            if e["endpoint_name"] == self.ENDPOINT:
                self.token = e["api_key"]
                self._save_token(self.token)
                return self.token
        raise RuntimeError(f"no endpoint named {self.ENDPOINT!r} on this account")

    @staticmethod
    def token_expiry(token):
        """Expiry (unix time) written inside the token, or None if it cannot be read."""
        try:
            outer = json.loads(base64.b64decode(token + "=" * (-len(token) % 4)))
            return float(json.loads(outer["msg"])["expiration"])
        except Exception:
            return None

    def _read_saved_token(self):
        try:
            with open(self.token_file, encoding="utf-8") as f:
                token = json.load(f)["token"]
        except (OSError, ValueError, KeyError):
            return None
        expiry = self.token_expiry(token)
        if expiry is not None and expiry - time.time() < 300:   # expired or about to
            return None
        return token

    def _save_token(self, token):
        # Owner-only file: the token lets anyone call the endpoint.
        fd = os.open(self.token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"endpoint": self.ENDPOINT, "token": token,
                       "expires": self.token_expiry(token), "saved_at": time.time()}, f)

    # ------------------------------------------------------------ step 2: worker

    def _route(self, cost):
        """Ask the router for a worker, polling until one is ready."""
        body = {"endpoint": self.ENDPOINT, "api_key": self.token, "cost": cost}
        deadline = time.time() + self.route_wait
        while True:
            status, answer = self._http_json(
                "POST", f"{ROUTER}/route/", body,
                headers={"Authorization": f"Bearer {self.token}"})
            if status == 200 and isinstance(answer, dict) and answer.get("url"):
                return answer
            text = json.dumps(answer, ensure_ascii=False) if not isinstance(answer, str) else answer
            if status in (401, 403) or "unauthorized" in text.lower():
                raise TokenRejected(text)
            if status != 200 or "error_msg" in text or time.time() > deadline:
                raise RuntimeError(f"route failed: HTTP {status}: {text}")
            body["request_idx"] = answer.get("request_idx", 0)   # keep our place in the queue
            time.sleep(1)

    # ------------------------------------------------------------ step 3: request

    def request(self, path, payload, cost):
        """Route and send one request; on a rejected token fetch a new one and retry once.

        Returns (HTTP status, decoded JSON or error text).
        """
        for attempt in (1, 2):
            try:
                auth = self._route(cost)
            except TokenRejected:
                if attempt == 2:
                    raise
                self.fetch_token(force=True)
                continue
            status, body = self._http_json(
                "POST", auth["url"] + path, {"auth_data": auth, "payload": payload},
                context=self.ssl_context, timeout=self.timeout)
            if status == 401 and attempt == 1:     # worker refused the signature
                self.fetch_token(force=True)
                continue
            self.last_worker = auth["url"]
            return status, body

    def diacritize(self, text, decisions=False):
        """Text in, the API's answer out: answer["text"] is what goes to the TTS."""
        status, body = self.request("/diacritize", {"text": text, "decisions": decisions},
                                    cost=max(len(text), 1))
        if status != 200:
            raise RuntimeError(f"/diacritize: HTTP {status}: {body}")
        return body

    def diacritize_batch(self, texts, decisions=False):
        """Several texts in one request; returns one answer per text, in order."""
        status, body = self.request("/diacritize/batch", {"texts": texts, "decisions": decisions},
                                    cost=max(sum(map(len, texts)), 1))
        if status != 200:
            raise RuntimeError(f"/diacritize/batch: HTTP {status}: {body}")
        return body["results"]

    # ------------------------------------------------------------ HTTP helpers

    @staticmethod
    def _http_json(method, url, body=None, headers=None, context=None, timeout=30):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=context) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(errors="replace")

    @staticmethod
    def _vast_ssl_context():
        """TLS context trusting Vast's CA, which signs the workers' certificates."""
        path = os.path.join(tempfile.gettempdir(), "jvastai_root.cer")
        if not os.path.exists(path):
            with urllib.request.urlopen(VAST_CA_URL, timeout=10) as r:
                pem = r.read()
            with open(path, "wb") as f:
                f.write(pem)
        return ssl.create_default_context(cafile=path)


# ---------------------------------------------------------------- tests

DEFAULT_TEXTS = [
    "إزيك عامل إيه النهارده",
    "صباح الخير يا جماعة، الجو حلو قوي",
    "تصبح على خير وإن شاء الله نتقابل بكره",
]


def check_result(r):
    assert isinstance(r, dict), f"not a JSON object: {r!r}"
    for field in ("text", "input", "marked", "ms"):
        assert field in r, f"missing field {field!r}: {r}"
    assert r["text"], "empty output text"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--text", action="append", help="text to diacritize (repeatable)")
    args = ap.parse_args()
    texts = args.text or DEFAULT_TEXTS

    failures = 0

    def run(name, fn):
        nonlocal failures
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failures += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")

    client = TashkeelVastClient()
    expiry = TashkeelVastClient.token_expiry(client.token)
    print(f"endpoint: {client.ENDPOINT}\ntoken saved in {client.token_file}, expires "
          f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(expiry)) if expiry else 'unknown'}")

    def test_single():
        for text in texts:
            t0 = time.time()
            r = client.diacritize(text, decisions=True)
            check_result(r)
            assert isinstance(r.get("decisions"), list), "decisions requested but missing"
            print(f"      {(time.time() - t0) * 1000:6.0f} ms  {client.last_worker}\n"
                  f"      in : {text}\n      out: {r['text']}  (marked {r['marked']})")

    def test_batch():
        t0 = time.time()
        results = client.diacritize_batch(texts)
        assert len(results) == len(texts), f"expected {len(texts)} results, got {len(results)}"
        for r in results:
            check_result(r)
        print(f"      {(time.time() - t0) * 1000:6.0f} ms  {len(results)} texts")

    def test_bad_request():
        status, body = client.request("/diacritize", {"decisions": True}, cost=1)
        assert status in (400, 422), f"missing text accepted: HTTP {status}: {body}"
        print(f"      HTTP {status} for a body without text (expected an error)")

    def test_token_refresh():
        # Plant a rejected token: the next call must replace it and still succeed.
        good = client.token
        client.token = "invalid-token"
        client._save_token(client.token)
        r = client.diacritize(texts[0])
        check_result(r)
        assert client.token == good, "the client did not fetch a new token"
        with open(client.token_file, encoding="utf-8") as f:
            assert json.load(f)["token"] == good, "the new token was not saved"
        print("      a rejected token was replaced, saved, and the request succeeded")

    run("diacritize()  POST /diacritize", test_single)
    run("diacritize_batch()  POST /diacritize/batch", test_batch)
    run("a body without text is rejected", test_bad_request)
    run("rejected token is refreshed and saved", test_token_refresh)

    print(f"\n{'all tests passed' if not failures else f'{failures} test(s) failed'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
