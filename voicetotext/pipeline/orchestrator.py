"""Threaded orchestration: audio source -> ASR -> translation -> callbacks."""
from __future__ import annotations

import queue
import threading
from typing import Callable

import numpy as np

from voicetotext.asr.engine import ASREngine
from voicetotext.asr.events import FinalTranscript, PartialTranscript
from voicetotext.audio.sources import AudioSource
from voicetotext.pipeline.messages import TranslatedLine
from voicetotext.translate.base import Translator
from voicetotext.translate.segmenter import split_sentences

_SENTINEL = object()


class Pipeline:
    def __init__(
        self,
        source: AudioSource,
        engine: ASREngine,
        translator: Translator,
        src_lang: str,
        tgt_lang: str,
        on_partial: Callable[[str], None],
        on_line: Callable[[TranslatedLine], None],
    ) -> None:
        self._source = source
        self._engine = engine
        self._translator = translator
        self._src = src_lang
        self._tgt = tgt_lang
        self._on_partial = on_partial
        self._on_line = on_line

        self._audio_q: queue.Queue = queue.Queue(maxsize=100)
        self._final_q: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._running = False

    def set_languages(self, src: str, tgt: str) -> None:
        self._src, self._tgt = src, tgt

    # ---- stage handlers (shared by threaded and blocking modes) ----
    def _handle_audio(self, chunk: np.ndarray) -> None:
        for ev in self._engine.accept(chunk):
            self._dispatch_event(ev)

    def _dispatch_event(self, ev) -> None:
        if isinstance(ev, PartialTranscript):
            self._on_partial(ev.text)
        elif isinstance(ev, FinalTranscript):
            self._translate_final(ev)

    def _translate_final(self, ev: FinalTranscript) -> None:
        for sentence in split_sentences(ev.text):
            translation = self._translator.translate(sentence, self._src, self._tgt)
            self._on_line(
                TranslatedLine(
                    source=sentence,
                    translation=translation,
                    t_start=ev.t_start,
                    t_end=ev.t_end,
                )
            )

    def _route(self, ev) -> None:
        """Threaded mode: partials go straight to the UI, finals to the MT queue."""
        if isinstance(ev, PartialTranscript):
            self._on_partial(ev.text)
        elif isinstance(ev, FinalTranscript):
            self._final_q.put(ev)

    # ---- blocking mode (file sources, tests) ----
    def run_file_blocking(self) -> None:
        self._source.start(self._handle_audio)
        for ev in self._engine.flush():
            self._dispatch_event(ev)

    # ---- threaded mode (live mic) ----
    def start(self) -> None:
        self._running = True

        def capture():
            self._source.start(lambda chunk: self._audio_q.put(chunk))
            self._audio_q.put(_SENTINEL)

        def asr():
            while True:
                item = self._audio_q.get()
                if item is _SENTINEL:
                    for ev in self._engine.flush():
                        self._route(ev)
                    self._final_q.put(_SENTINEL)
                    return
                for ev in self._engine.accept(item):
                    self._route(ev)

        def mt():
            while True:
                ev = self._final_q.get()
                if ev is _SENTINEL:
                    return
                self._translate_final(ev)

        self._threads = [
            threading.Thread(target=capture, daemon=True, name="capture"),
            threading.Thread(target=asr, daemon=True, name="asr"),
            threading.Thread(target=mt, daemon=True, name="mt"),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._running = False
        self._source.stop()
        for t in self._threads:
            t.join(timeout=5)
        self._threads = []
