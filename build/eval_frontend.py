"""Scores the frontend on the two homograph sets without the TTS, over a grid of policy
thresholds (the diacritizer runs once per item; Frontend.decide is applied per setting).

What the TTS reads for a word is the frontend's marking if it marks it, else the word's
default reading (the training text's most used one, what the TTS says plain). A reading
is judged against an item:

- authored (eval/homographs_authored.jsonl): judged on the letters that tell the item's
  reading from its other_reading (`essential`): "right", "wrong", or "undetermined" when
  the marking leaves those letters unmarked;
- mined (eval/homographs_mined.jsonl, references from the audio of held-out speech): on
  "rare" items (reference conflicts with the default) the share fixed; on "default"
  items the share broken (marked with a reading that conflicts with the reference).
  Mined items are capped per word, so rare readings are over-represented against
  running text: "net per 1,000 running ambiguous words" reweights them by the share of
  rare-reading occurrences in the test split (--rare-share, from the mining).

    .venv/bin/python build/eval_frontend.py --model-dir assets/diacritizer [--int8] [--out eval/results.json]
"""
import argparse
import collections
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from egy_frontend import Frontend  # noqa: E402
from egy_frontend.readings import DIFFERENT as BAD, conflict, letter_marks  # noqa: E402
from egy_frontend.text import SHADDA, canonicalize_marks, split_edges, strip_marks  # noqa: E402

MARKED = ("rare", "ambiguous", "differs")
GRID = {"deviate_ambiguous": [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01], "deviate_other": [0.9, 0.95, 0.99, 1.01]}


def word_at(fe: Frontend, text: str, index: int):
    plain = " ".join(strip_marks(t) for t in text.split(" "))
    _, decisions = fe.analyze(plain)
    core = split_edges(plain.split(" ")[index])[1]
    before = sum(1 for t in plain.split(" ")[:index] if split_edges(t)[1] == core)
    hits = [d for d in decisions if d.word == core]
    return (hits[before] if before < len(hits) else hits[-1]) if hits else None


def essential(reading: str, other: str | None) -> list[tuple[int, str]]:
    """The letters (index, mark of `reading`) that tell the two readings apart: those with
    a shadda on one only if there are any, else those with different vowels (a sukun
    counting as one). Judging on these alone keeps transcription conventions out (خِلِص
    and the Jais خَلَص are one word; the shadda of خَلَّص is what differs)."""
    la = letter_marks(reading)
    if not other:
        return [(i, m) for i, (_, m) in enumerate(la[:-1]) if m]
    lb = letter_marks(other)
    pairs = [(i, a, b) for i, ((_, a), (_, b)) in enumerate(zip(la[:-1], lb[:-1])) if a and b and a != b]
    sh = [(i, a) for i, a, b in pairs if (SHADDA in a) != (SHADDA in b)]
    return sh or [(i, a) for i, a, b in pairs]


def judge(r, reading, other) -> str:
    """right / wrong / undetermined: the heard marking r on the essential letters."""
    if not r:
        return "undetermined"
    lr = letter_marks(r)
    ess = essential(reading, other)
    if not ess:
        return "undetermined"
    by_shadda = other is not None and any((SHADDA in m) != (SHADDA in letter_marks(other)[i][1]) for i, m in ess)
    for i, m in ess:
        h = lr[i][1]
        if not h:
            return "undetermined"
        if by_shadda:
            if (SHADDA in h) != (SHADDA in m):
                return "wrong"
        elif h.replace(SHADDA, "") != m.replace(SHADDA, ""):
            return "wrong"
    return "right"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "assets" / "diacritizer"))
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--mined-limit", type=int, default=0, help="0 = all")
    ap.add_argument("--rare-share", type=float, default=0.21)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    fe = Frontend(model_dir=a.model_dir, quantized=a.int8)
    t0, n_sent = time.time(), 0

    def collect(path, index_of):
        rows = []
        for it in map(json.loads, path.open(encoding="utf-8")):
            d = word_at(fe, it["text"], index_of(it))
            if d is not None:
                rows.append((it, d, canonicalize_marks(d.predicted, fe.marks)))
        return rows

    authored = collect(ROOT / "eval" / "homographs_authored.jsonl",
                       lambda it: [split_edges(t)[1] for t in it["text"].split(" ")].index(it["target"]))
    mined_all = [json.loads(l) for l in (ROOT / "eval" / "homographs_mined.jsonl").open(encoding="utf-8")]
    step = max(1, len(mined_all) // a.mined_limit) if a.mined_limit else 1
    mined = []
    for it in mined_all[::step]:
        d = word_at(fe, it["text"], it["word_index"])
        if d is not None:
            mined.append((it, d, canonicalize_marks(d.predicted, fe.marks)))
    n_sent = len(authored) + len(mined)
    ms = 1000 * (time.time() - t0) / max(1, n_sent)

    def heard(d, canon, th):
        return canon if fe.decide(d.word, d.predicted, d.confidence, **th) in MARKED else d.default

    res = {"model_dir": a.model_dir, "int8": a.int8, "ms_per_sentence": round(ms, 1), "settings": []}
    plain_auth = collections.Counter(judge(d.default, it["reading"], it["other_reading"]) for it, d, _ in authored)
    diac_auth = collections.Counter(judge(c, it["reading"], it["other_reading"]) for it, d, c in authored)
    res["authored_plain_tts"] = dict(plain_auth)
    res["authored_diacritizer_alone"] = dict(diac_auth)
    agree = collections.Counter()
    for it, d, c in mined:
        k = it["kind"] + ("/" + it["conflict"] if it["conflict"] else "")
        agree[k + "/n"] += 1
        agree[k + "/ok"] += conflict(c, it["reference"]) not in BAD
    res["mined_diacritizer_agrees"] = {k[:-2]: round(agree[k[:-2] + "/ok"] / v, 3) for k, v in agree.items() if k.endswith("/n")}
    for va, vo in itertools.product(GRID["deviate_ambiguous"], GRID["deviate_other"]):
        th = {"deviate_ambiguous": va, "deviate_other": vo}
        au = collections.Counter(judge(heard(d, c, th), it["reading"], it["other_reading"]) for it, d, c in authored)
        m = collections.Counter()
        for it, d, c in mined:
            marked = fe.decide(d.word, d.predicted, d.confidence, **th) in MARKED
            if it["kind"] == "rare":
                m["rare"] += 1
                m["fixed"] += marked and conflict(c, it["reference"]) not in BAD
            else:
                m["default"] += 1
                m["broken"] += marked and conflict(c, it["reference"]) in BAD
        fixed, broken = m["fixed"] / max(1, m["rare"]), m["broken"] / max(1, m["default"])
        res["settings"].append({**th, "authored_right": round(au["right"] / len(authored), 3),
                                "authored_wrong": round(au["wrong"] / len(authored), 3),
                                "mined_fixed": round(fixed, 3), "mined_broken": round(broken, 3),
                                "net_per_1000": round(1000 * (a.rare_share * fixed - (1 - a.rare_share) * broken), 1)})
    n = len(authored)
    print(f"{a.model_dir}  int8={a.int8}  {ms:.1f} ms/sentence  authored n={n}, mined n={len(mined)}")
    print("authored, plain TTS:", {k: round(v / n, 3) for k, v in plain_auth.items()},
          " diacritizer alone:", {k: round(v / n, 3) for k, v in diac_auth.items()})
    print("mined, diacritizer agrees with reference:", res["mined_diacritizer_agrees"])
    print(f"{'dev_amb':>7} {'dev_oth':>7} {'auth_right':>10} {'auth_wrong':>10} {'fixed':>6} {'broken':>6} {'net/1000':>8}")
    for s in res["settings"]:
        print(f"{s['deviate_ambiguous']:7.2f} {s['deviate_other']:7.2f} {s['authored_right']:10.3f} {s['authored_wrong']:10.3f} "
              f"{s['mined_fixed']:6.3f} {s['mined_broken']:6.3f} {s['net_per_1000']:8.1f}")
    if a.out:
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
