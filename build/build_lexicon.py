"""Phase 1: the word lexicon of the TTS training text (assets/lexicon.json.gz).

The TTS models (Audio8, Pocket TTS) learned to read Egyptian from the transcripts in
--transcripts. What they say for a plain word is, in practice, the reading those
transcripts gave it most often, and a word they rarely saw has no reliable reading at
all. This records, for every letter skeleton of that text:

- uses: how often it occurs, plain or marked (the rare-word rule of the policy);
- readings: its marked uses grouped by reading, most used first, as [key, uses, form].
  The key is the word's marks without sukun and without the marks of its last letter,
  where the Jais transcripts are inconsistent (build_manifest.py cmd_lexicon in the TTS
  repo); the form is the reading's most used full marking (sukun kept, final marks
  dropped), what the frontend compares predictions with;
- forms: its 12 most used marked forms (>= MIN_USES uses), what canonicalize_marks
  rewrites a caller's marks to;
- known: skeletons with >= MIN_USES uses;
- ambiguous: the hint lexicon the TTS was trained with (--ambiguous), copied as is so
  the frontend marks exactly the words the training hint mode marked.

    python build/build_lexicon.py --transcripts /data/egy/v2/train_aligned.jsonl \\
        --ambiguous /data/egy/v2/ambiguous_words.json --out assets/lexicon.json.gz
"""
import argparse
import collections
import gzip
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egy_frontend.text import ARABIC_WORD, DIACRITICS, SUKUN, normalize_egyptian, split_edges, strip_marks  # noqa: E402

MIN_USES = 2
_FINAL_MARKS = re.compile(f"[{DIACRITICS}]+$")


def reading(core: str) -> str:
    return _FINAL_MARKS.sub("", core).replace(SUKUN, "")


def count(lines: list[str]):
    uses, forms, readings = collections.Counter(), collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for line in lines:
        for token in normalize_egyptian(json.loads(line)["transcript"]).split(" "):
            core = split_edges(token)[1]
            if not ARABIC_WORD.match(core):
                continue
            sk = strip_marks(core)
            uses[sk] += 1
            if core != sk:
                forms[sk][core] += 1
                r = reading(core)
                if r != sk:
                    readings[sk][(r, _FINAL_MARKS.sub("", core))] += 1
    return uses, forms, readings


def group(c: collections.Counter, top: int = 8) -> list[list]:
    """(key, form) counts -> [[key, uses, most used form], ...], most used key first."""
    keys, form = collections.Counter(), {}
    for (k, f), n in c.most_common():
        keys[k] += n
        form.setdefault(k, f)
    return [[k, n, form[k]] for k, n in keys.most_common(top)]


def shards(path: Path, size: int = 20000):
    buf = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            buf.append(line)
            if len(buf) == size:
                yield buf
                buf = []
    if buf:
        yield buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", type=Path, default=Path("/data/egy/v2/train_aligned.jsonl"))
    ap.add_argument("--ambiguous", type=Path, default=Path("/data/egy/v2/ambiguous_words.json"))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "assets" / "lexicon.json.gz")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    uses, forms, readings = collections.Counter(), collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    with Pool(a.workers) as pool:
        for u, f, r in pool.imap_unordered(count, shards(a.transcripts)):
            uses.update(u)
            for k, c in f.items():
                forms[k].update(c)
            for k, c in r.items():
                readings[k].update(c)
    ambiguous = json.loads(a.ambiguous.read_text())
    lex = {
        "source": str(a.transcripts),
        "min_uses": MIN_USES,
        "uses": dict(uses.most_common()),
        "readings": {k: group(c) for k, c in readings.items()},
        "forms": {k: [[x, n] for x, n in c.most_common(12) if n >= MIN_USES]
                  for k, c in forms.items() if c.most_common(1)[0][1] >= MIN_USES},
        "known": sorted(k for k, n in uses.items() if n >= MIN_USES),
        "ambiguous": ambiguous,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, "wt", encoding="utf-8") as fh:
        json.dump(lex, fh, ensure_ascii=False)
    total = sum(uses.values())
    report = {
        "tokens": total, "skeletons": len(uses), "marked_skeletons": len(readings),
        "known": len(lex["known"]), "ambiguous": len(ambiguous),
        "ambiguous_token_share": round(sum(uses[k] for k in ambiguous) / total, 3),
        **{f"skeletons_under_{t}_uses": sum(n < t for n in uses.values()) for t in (5, 20, 100)},
        **{f"token_share_under_{t}_uses": round(sum(n for n in uses.values() if n < t) / total, 4) for t in (5, 20, 100)},
    }
    (a.out.parent / "lexicon_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
