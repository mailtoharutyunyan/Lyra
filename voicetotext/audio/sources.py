"""Audio sources: file playback (dev/test) and live microphone."""
from __future__ import annotations

import time
import wave
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np

from voicetotext.audio.resample import TARGET_RATE, to_mono_16k

OnAudio = Callable[[np.ndarray], None]


def RMS(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


class AudioSource(Protocol):
    def start(self, on_audio: OnAudio) -> None: ...
    def stop(self) -> None: ...


class FileSource:
    streaming = False  # start() blocks until the whole file has been delivered

    def __init__(self, path: str | Path, chunk_ms: int = 100, realtime: bool = False) -> None:
        self._path = Path(path)
        self._chunk_ms = chunk_ms
        self._realtime = realtime
        self._stopped = False

    def start(self, on_audio: OnAudio) -> None:
        self._stopped = False
        with wave.open(str(self._path), "rb") as w:
            rate = w.getframerate()
            channels = w.getnchannels()
            width = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[width]
        data = np.frombuffer(raw, dtype=dtype)
        if channels > 1:
            data = data.reshape(-1, channels)
        mono16k = to_mono_16k(data, rate)

        step = int(TARGET_RATE * self._chunk_ms / 1000)
        for i in range(0, len(mono16k), step):
            if self._stopped:
                break
            chunk = mono16k[i : i + step]
            on_audio(chunk)
            if self._realtime:
                time.sleep(self._chunk_ms / 1000)

    def stop(self) -> None:
        self._stopped = True


class MicSource:
    streaming = True  # start() returns immediately; audio arrives via callback

    def __init__(self, device: Optional[int] = None, blocksize_ms: int = 100) -> None:
        self._device = device
        self._blocksize_ms = blocksize_ms
        self._stream = None

    def start(self, on_audio: OnAudio) -> None:
        import sounddevice as sd

        info = sd.query_devices(self._device, "input")
        native_rate = int(info["default_samplerate"])
        blocksize = int(native_rate * self._blocksize_ms / 1000)

        def _cb(indata, frames, time_info, status):  # runs on PortAudio thread
            chunk = to_mono_16k(indata.copy(), native_rate)  # copy: no work beyond convert
            on_audio(chunk)

        self._stream = sd.InputStream(
            samplerate=native_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            device=self._device,
            callback=_cb,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
