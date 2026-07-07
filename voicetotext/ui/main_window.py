"""Main application window: controls, transcript, level meter."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QMainWindow, QProgressBar, QPushButton,
    QVBoxLayout, QWidget,
)

from voicetotext.pipeline.messages import TranslatedLine
from voicetotext.translate.segmenter import LANG_CODES
from voicetotext.ui.transcript_view import TranscriptView

_TARGETS = ["eng_Latn", "rus_Cyrl", "ukr_Cyrl", "hye_Armn", "deu_Latn", "fra_Latn", "spa_Latn"]


class MainWindow(QMainWindow):
    # thread-safe bridges: pipeline worker threads emit these; slots run on GUI thread
    line_ready = Signal(object)
    partial_ready = Signal(str)
    level_ready = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VoiceToText")
        self.resize(700, 500)
        self._pipeline = None  # set by app wiring

        self.src_combo = QComboBox()
        self.src_combo.addItem("auto", "auto")
        for two, flores in LANG_CODES.items():
            self.src_combo.addItem(f"{two} ({flores})", flores)
        self.tgt_combo = QComboBox()
        for flores in _TARGETS:
            self.tgt_combo.addItem(flores, flores)
        self.tgt_combo.setCurrentText("rus_Cyrl")

        self.toggle_btn = QPushButton("Start")
        self.toggle_btn.clicked.connect(self.on_toggle)
        self.level = QProgressBar()
        self.level.setRange(0, 100)
        self.level.setTextVisible(False)

        controls = QHBoxLayout()
        controls.addWidget(self.src_combo)
        controls.addWidget(self.tgt_combo)
        controls.addWidget(self.toggle_btn)
        controls.addWidget(self.level, stretch=1)

        self.transcript = TranscriptView()

        root = QVBoxLayout()
        root.addLayout(controls)
        root.addWidget(self.transcript, stretch=1)
        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self.line_ready.connect(self._on_line)
        self.partial_ready.connect(self.transcript.set_partial)
        self.level_ready.connect(lambda v: self.level.setValue(int(min(1.0, v * 3) * 100)))

    def set_pipeline(self, pipeline) -> None:
        self._pipeline = pipeline

    def _on_line(self, line: TranslatedLine) -> None:
        self.transcript.add_line(line.source, line.translation, line.t_start, line.t_end)

    def on_toggle(self) -> None:
        if self._pipeline is None:
            return
        if self.toggle_btn.text() == "Start":
            self._pipeline.set_languages(self.src_combo.currentData(), self.tgt_combo.currentData())
            self._pipeline.start()
            self.toggle_btn.setText("Stop")
        else:
            self._pipeline.stop()
            self.toggle_btn.setText("Start")
