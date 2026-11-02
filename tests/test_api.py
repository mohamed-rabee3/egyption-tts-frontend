"""API tests. Skipped unless the exported model is in assets/diacritizer."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("EGY_MODEL_DIR", ROOT / "assets" / "diacritizer"))

pytest.importorskip("fastapi", reason="pip install -r requirements-api.txt")
if not (MODEL / "crf.npz").exists():
    pytest.skip("no exported model: HF_TOKEN=... python build/fetch_model.py",
                allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from api.config import Settings, cpu_budget  # noqa: E402
from api.main import create_app  # noqa: E402

SENTENCE = "البلكونة اتملت حمام!"


@pytest.fixture(scope="module")
def client():
    # fp32 is the production default and the only one the policy thresholds are tuned
    # for: the int8 encoder's confidences fall below them, so it leaves words plain.
    cfg = Settings.from_env()
    cfg = type(cfg)(**{**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                       "quantized": False, "workers": 2, "queue": 4, "cache": 64})
    with TestClient(create_app(cfg)) as c:   # the context manager runs lifespan
        yield c


def test_cpu_budget_is_sane():
    n = cpu_budget()
    assert 1 <= n <= (os.cpu_count() or 1)


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "healthy"
    assert client.get("/ready").json()["status"] == "ready"


def test_config_reports_capacity(client):
    cfg = client.get("/config").json()
    assert cfg["capacity"] == cfg["workers"] + cfg["queue"]
    assert cfg["model"] == "fp32"


def test_diacritize_marks_the_ambiguous_word(client):
    r = client.post("/diacritize", json={"text": SENTENCE})
    assert r.status_code == 200
    body = r.json()
    assert body["input"] == SENTENCE
    assert body["text"] != SENTENCE          # marks were added
    assert body["marked"] >= 1
    rules = {d["word"]: d["rule"] for d in body["decisions"]}
    assert rules["حمام"] == "ambiguous"
    assert body["ms"] > 0


def test_decisions_can_be_omitted(client):
    body = client.post("/diacritize", json={"text": SENTENCE, "decisions": False}).json()
    assert body["decisions"] is None
    assert body["text"]


def test_cache_marks_the_second_call(client):
    text = "الغسالة صفرت والفني جاي يركب التكييف."
    client.post("/diacritize", json={"text": text})
    assert client.post("/diacritize", json={"text": text}).json()["cached"] is True


def test_batch_matches_single(client):
    texts = [SENTENCE, "السنة الجديدة بتبدأ بكرة."]
    batch = client.post("/diacritize/batch", json={"texts": texts}).json()
    assert [r["input"] for r in batch["results"]] == texts
    for t, r in zip(texts, batch["results"]):
        assert client.post("/diacritize", json={"text": t}).json()["text"] == r["text"]


@pytest.mark.parametrize("payload,code", [
    ({"text": "   "}, 422),
    ({"text": ""}, 422),
    ({}, 422),
    ({"text": "ا" * 20_001}, 413),
])
def test_bad_input(client, payload, code):
    assert client.post("/diacritize", json=payload).status_code == code


def test_batch_limits(client):
    assert client.post("/diacritize/batch", json={"texts": []}).status_code == 422
    big = {"texts": [SENTENCE] * 1000}
    assert client.post("/diacritize/batch", json=big).status_code == 413


def test_stats_counts_requests(client):
    before = client.get("/stats").json()["served"]
    client.post("/diacritize", json={"text": "عندي موعد مهم بكرة الصبح"})
    after = client.get("/stats").json()
    assert after["served"] > before
    assert after["in_flight"] == 0
    assert after["failed"] == 0


def test_timing_header(client):
    assert float(client.get("/health").headers["x-process-time-ms"]) >= 0
