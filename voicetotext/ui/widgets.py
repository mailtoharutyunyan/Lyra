"""Custom widgets: an aurora level meter that visibly reacts to incoming audio."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath
from PySide6.QtWidgets import QWidget

_TEAL = QColor("#3DD6C4")
_VIOLET = QColor("#7C6CFF")
_TRACK = QColor("#232936")


class LevelMeter(QWidget):
    """Horizontal audio-level bar. Feed it RMS in [0, ~1]; it eases toward the value
    and decays when audio stops, so silence reads as an empty bar."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._target = 0.0
        self._shown = 0.0
        self.setMinimumHeight(10)
        self.setMinimumWidth(120)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)  # ~30 fps easing

    def set_level(self, rms: float) -> None:
        # RMS is small; scale so normal speech fills most of the bar.
        self._target = max(0.0, min(1.0, rms * 4.0))

    def _tick(self) -> None:
        # ease up fast, fall slower (VU-meter feel)
        if self._target > self._shown:
            self._shown += (self._target - self._shown) * 0.5
        else:
            self._shown += (self._target - self._shown) * 0.2
        self._target *= 0.85  # decay so it drops when audio stops
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        radius = h / 2

        track = QPainterPath()
        track.addRoundedRect(0, 0, w, h, radius, radius)
        p.fillPath(track, _TRACK)

        fill_w = max(h, self._shown * w)
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.0, _TEAL)
        grad.setColorAt(1.0, _VIOLET)
        clip = QPainterPath()
        clip.addRoundedRect(0, 0, fill_w, h, radius, radius)
        p.fillPath(clip, grad)
        p.end()
