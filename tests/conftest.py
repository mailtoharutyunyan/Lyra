import numpy as np
import pytest


@pytest.fixture
def speech_wav_16k():
    """Placeholder speech fixture.

    Replace the .npy load below with a real 16 kHz mono float32 recording of a
    known phrase for the integration test to assert on. Until then this fixture
    returns silence and the dependent integration test will simply not match.
    """
    from pathlib import Path
    p = Path(__file__).parent / "fixtures" / "speech_16k.npy"
    if p.exists():
        samples = np.load(p).astype(np.float32)
        return samples, "hello"  # expected substring for the recorded phrase
    return np.zeros(16000, dtype=np.float32), "hello"
