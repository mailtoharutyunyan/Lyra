import wave
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def speech_wav_16k():
    """Real 16 kHz mono speech fixture ("Hello, how are you today? ...").

    Generated with macOS `say` (see scripts/make_speech_fixture.sh). Returns
    (samples float32 16k, expected_substring). Skips if the fixture is absent.
    """
    p = Path(__file__).parent / "fixtures" / "speech_en_16k.wav"
    if not p.exists():
        pytest.skip("speech fixture not generated (run scripts/make_speech_fixture.sh)")
    with wave.open(str(p), "rb") as w:
        raw = w.readframes(w.getnframes())
    samples = (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)
    return samples, "how are you"


@pytest.fixture
def seamless_ready():
    """Provide (samples, expected) for the Seamless functional test, or skip.

    Uses the English fixture with source_lang='eng' to prove the SeamlessM4T
    engine transcribes end-to-end (macOS has no Armenian TTS voice to generate
    Armenian audio locally; the engine path is language-agnostic).
    """
    import importlib.util

    from voicetotext import config

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed (pip install voicetotext[seamless])")
    if not (config.models_dir() / "seamless" / ".complete").exists():
        pytest.skip("Seamless model not downloaded")
    p = Path(__file__).parent / "fixtures" / "speech_en_16k.wav"
    if not p.exists():
        pytest.skip("speech fixture not generated")
    with wave.open(str(p), "rb") as w:
        raw = w.readframes(w.getnframes())
    samples = (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)
    return samples, "how are you"
