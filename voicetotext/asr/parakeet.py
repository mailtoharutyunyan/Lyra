"""Parakeet TDT 0.6B v3 offline ASR with Silero VAD (simulated streaming)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from voicetotext.asr.events import Event, FinalTranscript, PartialTranscript
from voicetotext.audio.resample import TARGET_RATE
from voicetotext.models.download import ensure_model
from voicetotext.models.registry import PARAKEET, SILERO_VAD

_VAD_WINDOW = 512  # samples @ 16 kHz (Silero requirement)


class ParakeetEngine:
    def __init__(
        self,
        model_dir: str | Path,
        vad_model_path: str | Path,
        *,
        num_threads: int = 4,
        partial_interval_s: float = 0.4,
        min_silence_s: float = 0.15,
        max_speech_s: float = 8.0,
    ) -> None:
        import sherpa_onnx

        model_dir = Path(model_dir)
        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(next(model_dir.glob("encoder*.onnx"))),
            decoder=str(next(model_dir.glob("decoder*.onnx"))),
            joiner=str(next(model_dir.glob("joiner*.onnx"))),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=num_threads,
            model_type="nemo_transducer",
        )
        vad_cfg = sherpa_onnx.VadModelConfig()
        vad_cfg.silero_vad.model = str(vad_model_path)
        vad_cfg.silero_vad.threshold = 0.5
        vad_cfg.silero_vad.min_silence_duration = min_silence_s
        vad_cfg.silero_vad.min_speech_duration = 0.25
        vad_cfg.silero_vad.max_speech_duration = max_speech_s
        vad_cfg.sample_rate = TARGET_RATE
        self._vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=60)

        self._partial_interval = int(partial_interval_s * TARGET_RATE)
        self._buf = np.empty(0, dtype=np.float32)          # feeds VAD in 512-hops
        self._open_segment = np.empty(0, dtype=np.float32)  # accumulates current speech
        self._samples_since_partial = 0
        self._t = 0.0  # seconds consumed, for timestamps

    def _decode(self, samples: np.ndarray) -> str:
        stream = self._rec.create_stream()
        stream.accept_waveform(TARGET_RATE, samples)
        self._rec.decode_stream(stream)
        return stream.result.text.strip()

    def accept(self, samples: np.ndarray) -> list[Event]:
        events: list[Event] = []
        self._buf = np.concatenate([self._buf, samples.astype(np.float32)])

        while len(self._buf) >= _VAD_WINDOW:
            window = self._buf[:_VAD_WINDOW]
            self._buf = self._buf[_VAD_WINDOW:]
            self._t += _VAD_WINDOW / TARGET_RATE
            self._vad.accept_waveform(window)

            if self._vad.is_speech_detected():
                self._open_segment = np.concatenate([self._open_segment, window])
                self._samples_since_partial += _VAD_WINDOW
                if self._samples_since_partial >= self._partial_interval:
                    self._samples_since_partial = 0
                    text = self._decode(self._open_segment)
                    if text:
                        events.append(PartialTranscript(text=text))

            # drain any segments the VAD has closed
            while not self._vad.empty():
                seg = self._vad.front.samples
                self._vad.pop()
                text = self._decode(np.asarray(seg, dtype=np.float32))
                seg_dur = len(seg) / TARGET_RATE
                if text:
                    events.append(
                        FinalTranscript(text=text, t_start=self._t - seg_dur, t_end=self._t)
                    )
                self._open_segment = np.empty(0, dtype=np.float32)
                self._samples_since_partial = 0

        return events

    def flush(self) -> list[Event]:
        events: list[Event] = []
        self._vad.flush()
        while not self._vad.empty():
            seg = self._vad.front.samples
            self._vad.pop()
            text = self._decode(np.asarray(seg, dtype=np.float32))
            if text:
                seg_dur = len(seg) / TARGET_RATE
                events.append(
                    FinalTranscript(text=text, t_start=self._t - seg_dur, t_end=self._t)
                )
        self._open_segment = np.empty(0, dtype=np.float32)
        return events


def load_default() -> "ParakeetEngine":
    model_dir = ensure_model(PARAKEET)
    vad_dir = ensure_model(SILERO_VAD)
    vad_path = next(Path(vad_dir).glob("*silero_vad*.onnx"))
    return ParakeetEngine(model_dir, vad_path)
