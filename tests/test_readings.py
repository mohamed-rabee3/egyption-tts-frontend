import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egy_frontend.readings import conflict  # noqa: E402


def test_kinds():
    assert conflict("حَمَّام", "حَمَام") == "shadda"
    assert conflict("دَرْس", "دَرِّس") == "shadda"
    assert conflict("يِرْكَب", "يِرَكِّب") == "shadda"
    assert conflict("شَعْر", "شِعْر") == "vowel"
    assert conflict("رَجُل", "رِجْل") == "vowel"
    assert conflict("يِقُول", "يَقُول") == "initial"
    assert conflict("آثَارِهَا", "آثَارْهَا") == "sukun"
    assert conflict("حَمَّام", "حمَّام") is None
    assert conflict("كَتَبَ", "كَتَبُ") is None  # last letter ignored
