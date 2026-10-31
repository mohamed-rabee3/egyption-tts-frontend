"""Phase 2: train / validation / test splits for the Egyptian diacritizer.

The targets are the Jais-ASR diacritized transcripts the TTS models trained on, through
normalize_egyptian, so the diacritizer writes marks in the conventions the TTS learned.
gemini_synthetic is left out: its text is the plain prompt (5% of its words marked).

- Videos are assigned by hash (96 / 2 / 2), so no episode is in two splits; a sentence
  that occurs in two splits is dropped from both.
- Text is cut at word boundaries into chunks of at most 254 characters (the trainer's
  limit); tags are removed (not read by the diacritizer); a chunk without a marked word
  carries no label and is dropped.
- test_tts.jsonl: the TTS held-out set (valid_diacritized_aligned.jsonl, half of it
  speakers no split has).
- validation.jsonl is a 12,000-chunk sample, for fast checkpoint selection;
  validation_full.jsonl has all of it.

Line format, as the trainer reads it: {"target", "repo", "video", "chunk"}.

    python build/build_splits.py --out /data/tashkeel/egy_splits
"""
import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egy_frontend.text import TAGS, normalize_egyptian  # noqa: E402

EXCLUDE = {"gemini_synthetic"}
MAX_CHARS = 254
MARK = re.compile("[ًٌٍَُِّْٰ]")
TAG = re.compile("|".join(re.escape(t) for t in TAGS))


def video_of(path: str) -> str:
    """/data/egy/seg/<ds>/<video>_SPK<n>/<video>_<seg>.flac -> <ds>/<video>."""
    p = Path(path)
    return f"{p.parent.parent.name}/{re.sub(r'_SPK\d+$', '', p.parent.name)}"


def split_of(video: str) -> str:
    n = int(hashlib.sha256(("egy-tashkeel-v1:" + video).encode()).hexdigest()[:8], 16) % 100
    return "train" if n < 96 else "validation" if n < 98 else "test"


def chunks(text: str) -> list[str]:
    out, cur, length = [], [], 0
    for w in text.split(" "):
        if len(w) > MAX_CHARS:
            continue
        if cur and length + 1 + len(w) > MAX_CHARS:
            out.append(" ".join(cur))
            cur, length = [], 0
        length += len(w) + (1 if cur else 0)
        cur.append(w)
    if cur:
        out.append(" ".join(cur))
    return out


def prepare(text: str) -> str:
    return re.sub(r" +", " ", TAG.sub(" ", normalize_egyptian(text))).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", type=Path, default=Path("/data/egy/v2/train_aligned.jsonl"))
    ap.add_argument("--tts-heldout", type=Path, default=Path("/data/egy/v2/valid_diacritized_aligned.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("/data/tashkeel/egy_splits"))
    ap.add_argument("--validation-sample", type=int, default=12000)
    a = ap.parse_args()
    rows = collections.defaultdict(list)
    stats = collections.Counter()
    for line in a.transcripts.open(encoding="utf-8"):
        r = json.loads(line)
        if r["ds"] in EXCLUDE:
            stats["excluded_rows"] += 1
            continue
        video = video_of(r["path"])
        split = split_of(video)
        for i, t in enumerate(chunks(prepare(r["transcript"]))):
            if not MARK.search(t):
                stats["unlabelled_chunks"] += 1
                continue
            rows[split].append({"target": t, "repo": r["ds"], "video": video, "chunk": i})
    seen = collections.defaultdict(set)
    for split, rs in rows.items():
        for r in rs:
            seen[MARK.sub("", r["target"])].add(split)
    shared = {k for k, v in seen.items() if len(v) > 1}
    for split in rows:
        n = len(rows[split])
        rows[split] = [r for r in rows[split] if MARK.sub("", r["target"]) not in shared]
        stats[f"{split}_dropped_shared"] = n - len(rows[split])
    tts = [{"target": t, "repo": "tts_heldout/" + r["ds"], "video": video_of(r["path"]), "chunk": i}
           for r in map(json.loads, a.tts_heldout.open(encoding="utf-8"))
           for i, t in enumerate(chunks(prepare(r["transcript"]))) if MARK.search(t)]
    a.out.mkdir(parents=True, exist_ok=True)

    def dump(name, rs):
        with (a.out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[f"{name}_chunks"] = len(rs)

    random.Random(0).shuffle(rows["train"])
    dump("train", rows["train"])
    dump("validation_full", rows["validation"])
    dump("validation", random.Random(0).sample(rows["validation"], min(a.validation_sample, len(rows["validation"]))))
    dump("test", rows["test"])
    dump("test_tts", tts)
    for split in ("train", "validation", "test"):
        stats[f"{split}_videos"] = len({r["video"] for r in rows[split]})
    (a.out / "report.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
