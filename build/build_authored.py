"""Phase 2: eval/homographs_authored.jsonl from eval/authored_homographs.py.

One item per sentence, in the TTS repo's homographs.jsonl format (eval_homographs.py
scores it end to end through the TTS and a diacritizing ASR): id, text, target, sense,
reading, other_reading, checks. other_reading is the group's other reading, or for a
one-reading group the training text's default reading (what the TTS says plain).
Every sentence must contain the target as a word, alone or after a clitic prefix (ال،
و، ب، ف، ه...: the item's target is then that token, its readings the prefix + reading);
the build fails otherwise.
"""
import gzip
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from authored_homographs import GROUPS  # noqa: E402
from egy_frontend.readings import DIFFERENT, conflict, letter_marks  # noqa: E402
from egy_frontend.text import SHADDA, SUKUN, split_edges, strip_marks  # noqa: E402

PREFIXES = ["", "ال", "وال", "بال", "فال", "لل", "و", "ب", "ف", "ه", "وب", "وه"]
NAMES = {"َ": "fatha", "ُ": "damma", "ِ": "kasra", SUKUN: "sukun"}


def checks(reading: str, other: str) -> list[list[str]]:
    """The marks of `reading` that `other` does not share, as eval_homographs checks."""
    out = []
    for (ch, a), (_, b) in zip(letter_marks(reading)[:-1], letter_marks(other)[:-1]):
        if not a or not b:
            continue
        if (SHADDA in a) != (SHADDA in b):
            out.append([ch, "shadda" if SHADDA in a else "no_shadda"])
        elif a.replace(SHADDA, "") != b.replace(SHADDA, ""):
            out.append([ch, NAMES.get(a.replace(SHADDA, ""), "fatha")])
    return out


def main() -> None:
    lex = json.load(gzip.open(ROOT / "assets" / "lexicon.json.gz", "rt", encoding="utf-8"))
    items, bad = [], []
    for target, senses in GROUPS:
        for k, (sense, reading, sentences) in enumerate(senses):
            assert strip_marks(reading) == target, (target, reading)
            if len(senses) > 1:
                other = senses[1 - k][1] if len(senses) == 2 else next(s[1] for s in senses if s[1] != reading)
            else:
                rs = [r for _, _, r in lex["readings"].get(target, []) if conflict(r, reading) in DIFFERENT]
                other = rs[0] if rs else None
            for j, text in enumerate(sentences):
                text = unicodedata.normalize("NFC", text)
                words = [split_edges(t)[1] for t in text.split(" ")]
                pre = next((p for p in PREFIXES if p + target in words), None)
                if pre is None:
                    bad.append((target, text))
                    continue
                o = pre + other if other else None
                items.append({"id": f"{target}_{sense}_{j + 1}", "text": text, "target": pre + target, "sense": sense,
                              "reading": pre + reading, "other_reading": o,
                              "checks": checks(pre + reading, o) if o else []})
    if bad:
        for t, s in bad:
            print("target not a whole word:", t, "|", s)
        sys.exit(1)
    out = ROOT / "eval" / "homographs_authored.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(len(items), "items,", len(GROUPS), "words,", sum(len(s) for _, s in GROUPS), "readings ->", out)


if __name__ == "__main__":
    main()
