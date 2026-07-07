import pytest

from voicetotext.translate.base import Translator


def _model_available() -> bool:
    from voicetotext import config
    return (config.models_dir() / "nllb" / ".complete").exists()


def test_nllb_class_satisfies_protocol_without_loading():
    # import must not require the model
    from voicetotext.translate.nllb import NLLBTranslator
    assert hasattr(NLLBTranslator, "translate")


@pytest.mark.integration
@pytest.mark.skipif(not _model_available(), reason="NLLB model not downloaded")
def test_nllb_translates_english_to_russian():
    from voicetotext.translate.nllb import load_default
    tr = load_default()
    assert isinstance(tr, Translator)
    out = tr.translate("Hello, how are you?", "eng_Latn", "rus_Cyrl")
    assert isinstance(out, str) and out.strip()
    # crude sanity: Cyrillic characters present
    assert any("Ѐ" <= ch <= "ӿ" for ch in out)


@pytest.mark.integration
@pytest.mark.skipif(not _model_available(), reason="NLLB model not downloaded")
def test_nllb_empty_text_is_noop():
    from voicetotext.translate.nllb import load_default
    tr = load_default()
    assert tr.translate("   ", "eng_Latn", "rus_Cyrl") == ""
