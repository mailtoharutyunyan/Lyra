import numpy as np
import pytest

from voicetotext.asr.events import FinalTranscript, PartialTranscript


def _models_available() -> bool:
    from voicetotext import config
    return (
        (config.models_dir() / "parakeet" / ".complete").exists()
        and (config.models_dir() / "silero_vad" / ".complete").exists()
    )


def test_event_types_are_frozen():
    p = PartialTranscript(text="hi")
    f = FinalTranscript(text="hi.", t_start=0.0, t_end=1.0)
    with pytest.raises(Exception):
        p.text = "x"  # frozen
    assert f.t_end == 1.0


def test_engine_import_does_not_require_models():
    from voicetotext.asr.parakeet import ParakeetEngine
    assert hasattr(ParakeetEngine, "accept")
    assert hasattr(ParakeetEngine, "flush")


@pytest.mark.integration
@pytest.mark.skipif(not _models_available(), reason="Parakeet/VAD models not downloaded")
def test_silence_produces_no_finals():
    from voicetotext.asr.parakeet import load_default
    eng = load_default()
    silence = np.zeros(16000, dtype=np.float32)  # 1 s
    events = eng.accept(silence) + eng.flush()
    assert not any(isinstance(e, FinalTranscript) for e in events)


@pytest.mark.integration
@pytest.mark.skipif(not _models_available(), reason="Parakeet/VAD models not downloaded")
def test_speech_fixture_produces_a_final(speech_wav_16k):
    # speech_wav_16k fixture: (samples float32 16k, expected_substring)
    from voicetotext.asr.parakeet import load_default
    samples, expected = speech_wav_16k
    eng = load_default()
    events = []
    # feed in 100 ms chunks to exercise the streaming path
    for i in range(0, len(samples), 1600):
        events += eng.accept(samples[i : i + 1600])
    events += eng.flush()
    finals = [e for e in events if isinstance(e, FinalTranscript)]
    assert finals, "expected at least one final transcript"
    joined = " ".join(e.text.lower() for e in finals)
    assert expected.lower() in joined
