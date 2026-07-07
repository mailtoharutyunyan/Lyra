"""Translator interface — keeps the MT backend swappable (license-driven, see plan)."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Translator(Protocol):
    def translate(self, text: str, src: str, tgt: str) -> str:
        """Translate `text` from FLORES code `src` to FLORES code `tgt`."""
        ...
