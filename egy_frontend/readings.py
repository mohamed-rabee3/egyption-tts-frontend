"""Comparing two markings of one word: do they spell the same reading?

The Jais transcripts mark words partially and inconsistently (a vowel or a sukun left
out), so a missing mark is no information. Two markings of a skeleton are compared on
every letter both mark, but the last (whose marks are case endings, or inconsistent):

- "shadda": one has a shadda the other lacks (حَمَّام / حَمَام, دَرْس / دَرِّس);
- "vowel": different vowels (عَالَم / عَالِم, شَعْر / شِعْر);
- "sukun": a vowel against a sukun only: Egyptian vowel elision, one word said fuller or
  clipped (آثَارِهَا / آثَارْهَا, هَتِفْهَمُوا / هَتْفَهْمُوا). Also the rare real pair it
  cannot tell apart (لِبِس "wore" / لِبْس "clothes");
- "initial": different vowels on an imperfect prefix (ي ت ن أ as first letter) only,
  the prefix said the Egyptian or the formal way (يِقُول / يَقُول).

"shadda" and "vowel" are different words (DIFFERENT); "sukun" and "initial" are one
word said two ways. The strongest kind found is returned, None if the markings agree.
"""
from .text import DIACRITICS, SHADDA, SUKUN

DIFFERENT = ("shadda", "vowel")
PREFIX_LETTERS = set("يتنأ")
_RANK = {None: 0, "initial": 1, "sukun": 2, "vowel": 3, "shadda": 4}


def letter_marks(word: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for ch in word:
        if ch not in DIACRITICS:
            out.append((ch, ""))
        elif out:
            out[-1] = (out[-1][0], out[-1][1] + ch)
    return out


def conflict(a: str, b: str) -> str | None:
    """None if the two markings can be one reading, else the kind of difference."""
    la, lb = letter_marks(a), letter_marks(b)
    if [c for c, _ in la] != [c for c, _ in lb]:
        raise ValueError(f"different skeletons: {a} / {b}")
    kind = None
    for i, ((ch, ma), (_, mb)) in enumerate(zip(la[:-1], lb[:-1])):
        if not ma or not mb:
            continue
        if (SHADDA in ma) != (SHADDA in mb):
            return "shadda"
        va, vb = ma.replace(SHADDA, ""), mb.replace(SHADDA, "")
        if not va or not vb or va == vb:
            continue
        k = "sukun" if SUKUN in (va, vb) else "initial" if i == 0 and ch in PREFIX_LETTERS else "vowel"
        kind = max(kind, k, key=_RANK.get)
    return kind


def default_reading(readings: list[list]) -> str | None:
    """The reading the TTS gives the plain word: the training text's most used one (the
    lexicon's [key, uses, form] list), as its most used full marking."""
    return readings[0][2] if readings else None
