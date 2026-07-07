"""Messages passed between pipeline stages and to the UI."""
from __future__ import annotations

from dataclasses import dataclass

from voicetotext.asr.events import FinalTranscript, PartialTranscript  # re-export

Partial = PartialTranscript


@dataclass(frozen=True)
class TranslatedLine:
    source: str
    translation: str
    t_start: float
    t_end: float
