"""Needs assets/diacritizer (the exported model) and assets/lexicon.json.gz."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytestmark = pytest.mark.skipif(not (ROOT / "assets" / "diacritizer" / "crf.npz").exists(), reason="no exported model")


@pytest.fixture(scope="module")
def fe():
    from egy_frontend import Frontend
    return Frontend(threads=2)


def test_keeps_letters_tags_and_latin(fe):
    text = "[sighs] أنا بعت ال email للمدير امبارح الساعة 5."
    out = fe(text)
    assert out.startswith("[sighs] ")
    assert "email" in out and "5" in out
    from egy_frontend.text import strip_marks
    assert strip_marks(out) == strip_marks(fe(strip_marks(text)))


def test_caller_marks_win(fe):
    out, decisions = fe.analyze("البلكونة اتملت حَمَّام")
    assert [d.rule for d in decisions if d.word == "حَمَّام"] == ["caller"]


def test_long_text(fe):
    text = " ".join(["الولد راح المدرسة الصبح بدري وبعدين رجع البيت."] * 20)
    from egy_frontend.text import strip_marks
    assert strip_marks(fe(text)) == text
