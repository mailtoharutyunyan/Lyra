import pytest


def test_module_imports_without_torch():
    # seamless.py must import even when torch isn't installed (lazy imports).
    from voicetotext.asr import seamless
    assert hasattr(seamless.SeamlessEngine, "accept")
    assert hasattr(seamless.SeamlessEngine, "flush")


def test_language_mapping():
    from voicetotext.asr.seamless import to_seamless_lang
    assert to_seamless_lang("hye_Armn") == "hye"
    assert to_seamless_lang("hy") == "hye"
    assert to_seamless_lang("ru") == "rus"
    assert to_seamless_lang("eng") == "eng"  # already 3-letter
    with pytest.raises(KeyError):
        to_seamless_lang("zz")


def test_status_reports_fields():
    from voicetotext.asr.seamless import seamless_status
    st = seamless_status()
    assert set(st) >= {"available", "ram_gb", "has_torch", "notes"}
    assert isinstance(st["available"], bool)
    assert st["ram_gb"] > 0  # RAM probe works on this host


@pytest.mark.integration
def test_seamless_transcribes_armenian(seamless_ready):
    """Full functional test — needs torch + the ~5 GB model. Skips otherwise."""
    import numpy as np
    from voicetotext.asr.seamless import load_default
    from voicetotext.asr.events import FinalTranscript

    samples, expected_substr = seamless_ready
    eng = load_default(source_lang="eng")
    events = []
    for i in range(0, len(samples), 1600):
        events += eng.accept(samples[i:i + 1600])
    events += eng.flush()
    finals = [e for e in events if isinstance(e, FinalTranscript)]
    assert finals, "expected a transcript from Seamless"
    joined = " ".join(e.text.lower() for e in finals)
    assert expected_substr in joined
