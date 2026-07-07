"""Test doubles implementing the pipeline protocols (no ML, no hardware)."""
from __future__ import annotations

import numpy as np

from voicetotext.asr.events import Event, FinalTranscript, PartialTranscript


class FakeAudioSource:
    """Emits a fixed list of chunks synchronously, then returns."""
    def __init__(self, chunks: list[np.ndarray]) -> None:
        self._chunks = chunks
        self._stopped = False

    def start(self, on_audio):
        for c in self._chunks:
            if self._stopped:
                break
            on_audio(c)

    def stop(self):
        self._stopped = True


class ScriptedASREngine:
    """Returns pre-programmed events keyed by how many accept() calls have happened."""
    def __init__(self, script: dict[int, list[Event]], final_on_flush: list[Event] | None = None):
        self._script = script
        self._flush = final_on_flush or []
        self._n = 0

    def accept(self, samples) -> list[Event]:
        events = self._script.get(self._n, [])
        self._n += 1
        return list(events)

    def flush(self) -> list[Event]:
        return list(self._flush)


class RecordingTranslator:
    def __init__(self, mapping: dict[str, str] | None = None):
        self.calls: list[tuple[str, str, str]] = []
        self._map = mapping or {}

    def translate(self, text: str, src: str, tgt: str) -> str:
        self.calls.append((text, src, tgt))
        return self._map.get(text, f"[{tgt}]{text}")
