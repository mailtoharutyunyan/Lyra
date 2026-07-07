"""ASR engine interface."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from voicetotext.asr.events import Event


class ASREngine(Protocol):
    def accept(self, samples: np.ndarray) -> list[Event]:
        """Feed 16 kHz mono float32 samples; return any events produced now."""
        ...

    def flush(self) -> list[Event]:
        """Finalize any open segment (e.g. at end of a file)."""
        ...
