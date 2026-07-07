"""Scrolling transcript with export to plain text and SRT."""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget, QLabel


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


class TranscriptView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._lines: list[_Line] = []
        self._partial = ""
        self._list = QListWidget()
        self._partial_label = QLabel("")
        self._partial_label.setStyleSheet("color: gray; font-style: italic;")
        layout = QVBoxLayout(self)
        layout.addWidget(self._list, stretch=1)
        layout.addWidget(self._partial_label)

    def add_line(self, source: str, translation: str, t_start: float = 0.0, t_end: float = 0.0) -> None:
        self._lines.append(_Line(source, translation, t_start, t_end))
        item = QListWidgetItem(f"{source}\n    → {translation}")
        self._list.addItem(item)
        self._list.scrollToBottom()
        self.set_partial("")  # a finalized line clears the live row

    def set_partial(self, text: str) -> None:
        self._partial = text
        self._partial_label.setText(text)

    def current_partial(self) -> str:
        return self._partial

    def clear(self) -> None:
        self._lines.clear()
        self._list.clear()
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
