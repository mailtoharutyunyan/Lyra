"""Transcript events emitted by an ASR engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class PartialTranscript:
    text: str


@dataclass(frozen=True)
class FinalTranscript:
    text: str
    t_start: float
    t_end: float


Event = Union[PartialTranscript, FinalTranscript]
