"""Phase 2: the mined homograph test (eval/homographs_mined.jsonl).

Every occurrence, in the held-out test split, of a word of the hint lexicon whose
reference marking (Jais, from the audio) is compared with the word's default reading
(the training text's most used one, what the TTS says for it plain):

- "rare": the reference is a different word from the default (readings.DIFFERENT): the
  TTS would misread the plain word here. Diacritizer recall on these is the headline
  number;
- "default": the reference agrees with it, or is it said another way (elision, prefix
  vowel): the control (a frontend that marks these wrongly makes things worse).

Each item: {"id", "text" (the chunk, marked), "word_index", "skeleton", "reference",
"default", "kind" (rare / default), "conflict" (shadda / vowel, for rare)}. At most
--per-word items of each kind per skeleton, so frequent words do not dominate.

    python build/mine_homographs.py --split /data/tashkeel/egy_splits/test.jsonl
"""
import argparse
import collections
import gzip
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from egy_frontend.readings import DIFFERENT, conflict, default_reading  # noqa: E402
from egy_frontend.text import ARABIC_WORD, split_edges, strip_marks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=Path("/data/tashkeel/egy_splits/test.jsonl"))
    ap.add_argument("--lexicon", type=Path, default=ROOT / "assets" / "lexicon.json.gz")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "homographs_mined.jsonl")
    ap.add_argument("--per-word", type=int, default=3)
    a = ap.parse_args()
    lex = json.load(gzip.open(a.lexicon, "rt", encoding="utf-8"))
    ambiguous, readings = set(lex["ambiguous"]), lex["readings"]
    pools = collections.defaultdict(list)
    for n, line in enumerate(a.split.open(encoding="utf-8")):
        r = json.loads(line)
        for i, token in enumerate(r["target"].split(" ")):
            core = split_edges(token)[1]
            sk = strip_marks(core)
            if sk not in ambiguous or core == sk or not ARABIC_WORD.match(core):
                continue
            d = default_reading(readings.get(sk, []))
            if d is None:
                continue
            c = conflict(core, d)
            c = c if c in DIFFERENT else None
            pools[(sk, "rare" if c else "default")].append(
                {"text": r["target"], "word_index": i, "skeleton": sk, "reference": core, "default": d,
                 "kind": "rare" if c else "default", "conflict": c, "repo": r["repo"]})
    rng = random.Random(0)
    items = []
    for key in sorted(pools):
        items += rng.sample(pools[key], min(a.per_word, len(pools[key])))
    for k, it in enumerate(items):
        it["id"] = f"m{k:05d}"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps({"id": it.pop("id"), **it}, ensure_ascii=False) + "\n")
    c = collections.Counter((it["kind"], it["conflict"]) for it in items)
    print(json.dumps({"items": len(items), "skeletons": len({it["skeleton"] for it in items}),
                      **{f"{k}/{v}": n for (k, v), n in sorted(c.items(), key=str)}}, indent=1))


if __name__ == "__main__":
    main()
