"""macOS system-audio capture via the macloop engine (CoreAudio/ScreenCaptureKit).

macloop delivers 16 kHz mono float32 straight from its AsrSink, matching the
pipeline audio contract, so no resampling is needed here.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np

from voicetotext.audio.resample import TARGET_RATE

OnAudio = Callable[[np.ndarray], None]

_CHUNK_FRAMES = 1600  # 100 ms @ 16 kHz


class MacloopSystemSource:
    """Capture whatever the Mac is playing (system audio) and feed it to `on_audio`."""

    streaming = True  # start() returns immediately; audio arrives via callback

    def __init__(self, chunk_frames: int = _CHUNK_FRAMES) -> None:
        self._chunk_frames = chunk_frames
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self, on_audio: OnAudio) -> None:
        import macloop

        # Permission check is safe from the calling thread.
        if not macloop.screen_capture_access(prompt=True):
            raise PermissionError(
                "Screen & System Audio Recording permission is required to capture "
                "system audio. Grant it in System Settings > Privacy & Security."
            )

        self._running = True

        def _loop() -> None:
            # macloop's engine/sink are unsendable (pyo3): create, use, and close
            # them all on this one thread.
            engine = macloop.AudioEngine()
            try:
                stream = engine.create_stream(macloop.SystemAudioSource)
                route = engine.route(stream=stream)
                sink = macloop.AsrSink(
                    routes=[route],
                    chunk_frames=self._chunk_frames,
                    sample_rate=TARGET_RATE,
                    channels=1,
                    sample_format="f32",
                )
                try:
                    for chunk in sink.chunks():
                        if not self._running:
                            break
                        on_audio(np.asarray(chunk.samples, dtype=np.float32))
                finally:
                    sink.close()
            finally:
                engine.close()

        self._thread = threading.Thread(target=_loop, daemon=True, name="macloop-sys")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
