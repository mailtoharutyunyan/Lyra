import pytest

from voicetotext.translate.segmenter import LANG_CODES, split_sentences, to_flores


def test_splits_on_terminators_keeping_punctuation():
    out = split_sentences("Hello there. How are you? I am fine!")
    assert out == ["Hello there.", "How are you?", "I am fine!"]


def test_trailing_fragment_without_terminator_is_kept():
    out = split_sentences("This is unfinished")
    assert out == ["This is unfinished"]


def test_ellipsis_and_extra_whitespace():
    out = split_sentences("Wait…  really?   ")
    assert out == ["Wait…", "really?"]


def test_empty_input_yields_empty_list():
    assert split_sentences("   ") == []


def test_lang_codes_cover_required_languages():
    for two in ("en", "ru", "uk", "de", "fr", "es"):
        assert two in LANG_CODES
    assert LANG_CODES["en"] == "eng_Latn"
    assert LANG_CODES["ru"] == "rus_Cyrl"


def test_to_flores_accepts_two_letter_and_flores():
    assert to_flores("en") == "eng_Latn"
    assert to_flores("hye_Armn") == "hye_Armn"  # already FLORES, passthrough
    with pytest.raises(KeyError):
        to_flores("zz")
