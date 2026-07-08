"""Optional extended-language ASR via SeamlessM4T v2 (e.g. Armenian speech input).

Parakeet covers 25 European languages but not Armenian. SeamlessM4T v2 transcribes
~100 source languages. It is heavy (~5 GB, ~6 GB RAM, needs torch) and slow on CPU,
so it runs in "delayed captions" mode: VAD-segmented, finals only, no live partials.

torch/transformers are imported lazily so the base app stays torch-free; this engine
is an optional pack (`pip install voicetotext[seamless]`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from voicetotext.asr.events import Event, FinalTranscript
from voicetotext.audio.resample import TARGET_RATE
from voicetotext.models.download import ensure_model
from voicetotext.models.registry import SEAMLESS, SILERO_VAD

_VAD_WINDOW = 512

# Parakeet/FLORES-ish 2-letter -> Seamless 3-letter source codes (the languages a
# user is most likely to pick for the extended pack).
SEAMLESS_LANG = {
    "hy": "hye", "hye_Armn": "hye",
    "ru": "rus", "rus_Cyrl": "rus",
    "en": "eng", "eng_Latn": "eng",
    "ka": "kat", "kat_Geor": "kat",
    "fa": "pes", "az": "azj",
}


def _total_ram_gb() -> float:
    import os

    try:  # POSIX (macOS, Linux)
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)
    except (ValueError, AttributeError, OSError):
        pass
    try:  # Windows
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullTotalPhys / (1024 ** 3)
    except Exception:
        return 0.0


def seamless_status() -> dict:
    """Report whether the Seamless pack can run here (RAM + torch installed)."""
    import importlib.util

    from voicetotext.models.registry import SEAMLESS_MIN_RAM_GB

    ram = _total_ram_gb()
    has_torch = importlib.util.find_spec("torch") is not None
    enough_ram = ram >= SEAMLESS_MIN_RAM_GB
    ok = has_torch and enough_ram
    if ok:
        notes = "Seamless pack can run."
    elif not enough_ram:
        notes = f"Needs >= {SEAMLESS_MIN_RAM_GB} GB RAM (have {ram:.0f} GB)."
    else:
        notes = "Install the pack: pip install voicetotext[seamless]."
    return {"available": ok, "ram_gb": round(ram, 1), "has_torch": has_torch, "notes": notes}


def to_seamless_lang(code: str) -> str:
    if code in SEAMLESS_LANG:
        return SEAMLESS_LANG[code]
    if len(code) == 3:  # already a Seamless code
        return code
    raise KeyError(f"No Seamless language mapping for {code!r}")


class SeamlessEngine:
    """VAD-segmented offline transcription with SeamlessM4T v2 (finals only)."""

    def __init__(self, model_dir, vad_model_path, source_lang: str = "hye",
                 target_lang: str | None = None,
                 *, max_speech_s: float = 8.0, min_silence_s: float = 0.2) -> None:
        import sherpa_onnx
        import torch  # noqa: F401  (ensures the optional pack is installed)
        from transformers import AutoProcessor, SeamlessM4Tv2Model

        # Two modes:
        #   explicit source  -> transcribe in that language (ASR); NLLB translates after.
        #   source == "auto"  -> translate speech straight to the target (S2TT); Seamless
        #                        auto-detects the spoken language, no MT step needed.
        self.translates_directly = source_lang == "auto"
        gen_lang = target_lang if self.translates_directly else source_lang
        self._gen_lang = to_seamless_lang(gen_lang or "eng_Latn")
        self._processor = AutoProcessor.from_pretrained(str(model_dir))
        self._model = SeamlessM4Tv2Model.from_pretrained(str(model_dir))
        self._model.eval()

        vad_cfg = sherpa_onnx.VadModelConfig()
        vad_cfg.silero_vad.model = str(vad_model_path)
        vad_cfg.silero_vad.threshold = 0.5
        vad_cfg.silero_vad.min_silence_duration = min_silence_s
        vad_cfg.silero_vad.min_speech_duration = 0.25
        vad_cfg.silero_vad.max_speech_duration = max_speech_s
        vad_cfg.sample_rate = TARGET_RATE
        self._vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=60)
        self._buf = np.empty(0, dtype=np.float32)
        self._t = 0.0

    def _transcribe(self, samples: np.ndarray) -> str:
        import torch

        inputs = self._processor(
            audio=samples, sampling_rate=TARGET_RATE, return_tensors="pt"
        )
        with torch.no_grad():
            tokens = self._model.generate(
                **inputs, tgt_lang=self._gen_lang, generate_speech=False,
                no_repeat_ngram_size=3,  # curb the model's tendency to loop on music
            )
        # generate() returns token ids; decode the text stream
        seq = tokens[0].tolist()[0] if hasattr(tokens[0], "tolist") else tokens[0]
        return self._processor.decode(seq, skip_special_tokens=True).strip()

    def _drain_finals(self) -> list[Event]:
        events: list[Event] = []
        while not self._vad.empty():
            seg = self._vad.front.samples
            self._vad.pop()
            text = self._transcribe(np.asarray(seg, dtype=np.float32))
            if text:
                dur = len(seg) / TARGET_RATE
                events.append(FinalTranscript(text=text, t_start=self._t - dur, t_end=self._t))
        return events

    def accept(self, samples: np.ndarray) -> list[Event]:
        events: list[Event] = []
        self._buf = np.concatenate([self._buf, samples.astype(np.float32)])
        while len(self._buf) >= _VAD_WINDOW:
            window = self._buf[:_VAD_WINDOW]
            self._buf = self._buf[_VAD_WINDOW:]
            self._t += _VAD_WINDOW / TARGET_RATE
            self._vad.accept_waveform(window)
            events += self._drain_finals()
        return events

    def flush(self) -> list[Event]:
        self._vad.flush()
        return self._drain_finals()


def load_default(source_lang: str = "hye", target_lang: str | None = None) -> "SeamlessEngine":
    model_dir = ensure_model(SEAMLESS)
    vad_dir = ensure_model(SILERO_VAD)
    vad_path = next(Path(vad_dir).glob("*silero_vad*.onnx"))
    return SeamlessEngine(model_dir, vad_path, source_lang=source_lang, target_lang=target_lang)
