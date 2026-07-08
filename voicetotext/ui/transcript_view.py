"""Scrolling transcript history with a clean two-line look, plus .srt/.txt export."""
from __future__ import annotations

import html
from dataclasses import dataclass

from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget


@dataclass
class _Line:
    source: str
    translation: str
    t_start: float = 0.0
    t_end: float = 0.0


def _srt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


_DOC_CSS = """
<style>
  .row { margin: 0 0 14px 0; }
  .tr  { color: #F1F3F6; font-size: 15px; line-height: 140%; }
  .src { color: #7C8493; font-size: 12px; margin-top: 2px; }
</style>
"""


class TranscriptView(QWidget):
    """Read-only history: each utterance shows the translation, then the original
    beneath it in a muted tone — easy to scan."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._lines: list[_Line] = []
        self._partial = ""
        self._view = QTextEdit()
        self._view.setReadOnly(True)
        self._view.setFrameStyle(0)
        self._view.setStyleSheet(
            "QTextEdit { background: #14171D; border: 1px solid #232833; "
            "border-radius: 12px; padding: 12px 14px; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

    def _render(self) -> None:
        rows = []
        for ln in self._lines:
            rows.append(
                f'<div class="row"><div class="tr">{html.escape(ln.translation)}</div>'
                f'<div class="src">{html.escape(ln.source)}</div></div>')
        self._view.setHtml(_DOC_CSS + "".join(rows))
        sb = self._view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def add_line(self, source: str, translation: str, t_start: float = 0.0, t_end: float = 0.0) -> None:
        self._lines.append(_Line(source, translation, t_start, t_end))
        self._render()
        self.set_partial("")

    def set_partial(self, text: str) -> None:
        self._partial = text

    def current_partial(self) -> str:
        return self._partial

    def clear(self) -> None:
        self._lines.clear()
        self._render()
        self.set_partial("")

    def to_text(self) -> str:
        return "\n".join(f"{ln.source}\n{ln.translation}" for ln in self._lines)

    def to_srt(self) -> str:
        blocks = []
        for i, ln in enumerate(self._lines, start=1):
            blocks.append(
                f"{i}\n{_srt_ts(ln.t_start)} --> {_srt_ts(ln.t_end)}\n"
                f"{ln.source}\n{ln.translation}\n"
            )
        return "\n".join(blocks)
