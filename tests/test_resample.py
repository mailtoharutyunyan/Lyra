import numpy as np
import pytest

from voicetotext.audio.resample import TARGET_RATE, to_mono_16k


def test_target_rate_is_16k():
    assert TARGET_RATE == 16000


def test_downmix_stereo_to_mono():
    stereo = np.zeros((100, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    stereo[:, 1] = -0.5
    out = to_mono_16k(stereo, 16000)
    assert out.ndim == 1
    assert out.shape == (100,)
    assert np.allclose(out, 0.0, atol=1e-6)


def test_int16_is_scaled_to_unit_float():
    ints = np.array([32767, -32768, 0], dtype=np.int16)
    out = to_mono_16k(ints, 16000)
    assert out.dtype == np.float32
    assert out[0] == pytest.approx(1.0, abs=1e-3)
    assert out[1] == pytest.approx(-1.0, abs=1e-3)
    assert out[2] == pytest.approx(0.0)


def test_resample_changes_length_proportionally():
    # 1 second of 48 kHz mono -> ~16000 samples at 16 kHz
    src = np.zeros(48000, dtype=np.float32)
    out = to_mono_16k(src, 48000)
    assert abs(len(out) - 16000) < 50


def test_passthrough_when_already_16k_mono():
    src = np.linspace(-1, 1, 16000, dtype=np.float32)
    out = to_mono_16k(src, 16000)
    assert out.dtype == np.float32
    assert len(out) == 16000
