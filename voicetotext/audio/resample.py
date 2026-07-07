"""Convert arbitrary PCM audio to the pipeline contract: 16 kHz mono float32."""
from __future__ import annotations

import numpy as np
import soxr

TARGET_RATE = 16000


def _to_float32(samples: np.ndarray) -> np.ndarray:
    if samples.dtype == np.float32:
        return samples
    if samples.dtype == np.float64:
        return samples.astype(np.float32)
    if samples.dtype == np.int16:
        return (samples.astype(np.float32) / 32768.0)
    if samples.dtype == np.int32:
        return (samples.astype(np.float32) / 2147483648.0)
    # last resort: assume already float-like
    return samples.astype(np.float32)


def to_mono_16k(samples: np.ndarray, src_rate: int) -> np.ndarray:
    samples = np.asarray(samples)
    # Scale to float BEFORE downmixing: mean(axis=1) on an int array upcasts to
    # float64 first, which would make _to_float32 skip the integer-scaling branch
    # and leave multi-channel int16/int32 audio unscaled (outside [-1, 1]).
    data = _to_float32(samples)
    if data.ndim == 2:  # (frames, channels) -> mono
        data = data.mean(axis=1)
    if src_rate != TARGET_RATE:
        data = soxr.resample(data, src_rate, TARGET_RATE)
    return np.ascontiguousarray(data, dtype=np.float32)
