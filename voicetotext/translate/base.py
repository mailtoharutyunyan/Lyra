"""Translator interface — keeps the MT backend swappable (license-driven, see plan)."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Translator(Protocol):
    def translate(self, text: str, src: str, tgt: str) -> str:
        """Translate `text` from FLORES code `src` to FLORES code `tgt`."""
        ...


class PassthroughTranslator:
    """Returns text unchanged. Used when the ASR engine already emits the target
    language (SeamlessM4T speech-to-text-translation), so no MT step is needed."""

    def translate(self, text: str, src: str, tgt: str) -> str:
        return text
