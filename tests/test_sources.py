import wave

import numpy as np

from voicetotext.audio.sources import FileSource, RMS


def _write_wav(path, samples_i16, rate):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples_i16.tobytes())


def test_rms_of_silence_is_zero_and_signal_positive():
    assert RMS(np.zeros(100, dtype=np.float32)) == 0.0
    assert RMS(np.full(100, 0.5, dtype=np.float32)) > 0.4


def test_file_source_delivers_all_audio_as_16k_mono(tmp_path):
    rate = 48000
    samples = (np.sin(np.linspace(0, 100, rate)) * 10000).astype(np.int16)  # 1 s
    wav = tmp_path / "t.wav"
    _write_wav(wav, samples, rate)

    got = []
    src = FileSource(wav, chunk_ms=100, realtime=False)
    src.start(lambda chunk: got.append(chunk))

    total = np.concatenate(got)
    assert total.dtype == np.float32
    assert abs(len(total) - 16000) < 200      # ~1 s resampled to 16 kHz
    assert all(c.ndim == 1 for c in got)


def test_mic_source_imports():
    from voicetotext.audio.sources import MicSource
    assert hasattr(MicSource, "start") and hasattr(MicSource, "stop")
