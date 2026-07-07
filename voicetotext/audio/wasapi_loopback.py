"""Windows system-audio capture via WASAPI loopback (PyAudioWPatch).

Captures whatever the default output device is playing. No driver or permission
needed. Not functionally testable on non-Windows hosts; import is lazy.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np

from voicetotext.audio.resample import to_mono_16k

OnAudio = Callable[[np.ndarray], None]


class WasapiLoopbackSource:
    streaming = True  # start() returns immediately; audio arrives via callback

    def __init__(self, chunk_ms: int = 100) -> None:
        self._chunk_ms = chunk_ms
        self._pa = None
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self, on_audio: OnAudio) -> None:
        import pyaudiowpatch as pyaudio

        self._pa = pyaudio.PyAudio()
        loopback = self._pa.get_default_wasapi_loopback()
        rate = int(loopback["defaultSampleRate"])
        channels = int(loopback["maxInputChannels"])
        frames = int(rate * self._chunk_ms / 1000)
        self._running = True

        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            frames_per_buffer=frames,
            input=True,
            input_device_index=loopback["index"],
        )

        def _loop() -> None:
            while self._running:
                raw = self._stream.read(frames, exception_on_overflow=False)
                data = np.frombuffer(raw, dtype=np.float32)
                if channels > 1:
                    data = data.reshape(-1, channels)
                on_audio(to_mono_16k(data, rate))

        self._thread = threading.Thread(target=_loop, daemon=True, name="wasapi-loopback")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa is not None:
            self._pa.terminate()
        self._stream = self._pa = self._thread = None
