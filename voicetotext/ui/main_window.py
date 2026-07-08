"""Lyra main window: source + language pickers, a cinematic live caption, history."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QVBoxLayout, QWidget,
)

from voicetotext.pipeline.messages import TranslatedLine
from voicetotext.ui.languages import SOURCE_CODES, TARGET_CODES, name_for
from voicetotext.ui.transcript_view import TranscriptView
from voicetotext.ui.widgets import LevelMeter

_QSS = """
QWidget { background: #0E1116; color: #E8EAED;
    font-family: "SF Pro Text", "Avenir Next", "Segoe UI", sans-serif; font-size: 14px; }
QLabel#brand { font-family: "SF Pro Display", "Avenir Next", sans-serif;
    font-size: 20px; font-weight: 600; color: #E8EAED; }
QLabel#brandDot { color: #3DD6C4; font-size: 20px; font-weight: 700; }
QComboBox { background: #1A1F29; border: 1px solid #2A303C; border-radius: 8px;
    padding: 6px 10px; min-height: 20px; }
QComboBox:hover { border-color: #3DD6C4; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView { background: #1A1F29; border: 1px solid #2A303C;
    selection-background-color: #3DD6C4; selection-color: #0E1116; outline: none; }
QLabel#arrow { color: #6B7486; font-size: 16px; }
QPushButton#primary { background: #3DD6C4; color: #08201D; border: none;
    border-radius: 9px; padding: 8px 22px; font-weight: 700; }
QPushButton#primary:hover { background: #57E3D2; }
QPushButton#primary[recording="true"] { background: #FF6B6B; color: #2A0B0B; }
QPushButton#ghost { background: transparent; color: #8A93A3; border: 1px solid #2A303C;
    border-radius: 8px; padding: 6px 14px; }
QPushButton#ghost:hover { color: #E8EAED; border-color: #3DD6C4; }
QLabel#status { color: #8A93A3; font-size: 13px; }
QLabel#hero { font-family: "SF Pro Display", "Avenir Next", sans-serif;
    font-size: 30px; font-weight: 600; color: #F2F4F7; line-height: 130%; }
QLabel#heard { color: #7F8AA0; font-size: 16px; font-style: italic; }
QFrame#card { background: #12151C; border: 1px solid #1E2430; border-radius: 16px; }
"""


class MainWindow(QMainWindow):
    line_ready = Signal(object)
    partial_ready = Signal(str)
    level_ready = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Lyra")
        self.resize(860, 620)
        self.setStyleSheet(_QSS)
        self._make_pipeline = None
        self._pipeline = None
        self._file_path = None

        # --- control row ---
        brand = QLabel("◆")
        brand.setObjectName("brandDot")
        title = QLabel("Lyra")
        title.setObjectName("brand")

        self.source_combo = QComboBox()
        for label, kind in [("🎤  Microphone", "mic"),
                            ("🔊  System audio", "system"),
                            ("📂  Open file…", "file")]:
            self.source_combo.addItem(label, kind)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)

        self.src_combo = QComboBox()
        for code in SOURCE_CODES:
            self.src_combo.addItem(name_for(code), code)
        self.tgt_combo = QComboBox()
        for code in TARGET_CODES:
            self.tgt_combo.addItem(name_for(code), code)
        self.tgt_combo.setCurrentIndex(TARGET_CODES.index("rus_Cyrl"))

        arrow = QLabel("→")
        arrow.setObjectName("arrow")

        self.toggle_btn = QPushButton("Start")
        self.toggle_btn.setObjectName("primary")
        self.toggle_btn.clicked.connect(self.on_toggle)

        self.level = LevelMeter()

        controls = QHBoxLayout()
        controls.setSpacing(10)
        controls.addWidget(brand)
        controls.addWidget(title)
        controls.addSpacing(14)
        controls.addWidget(self.source_combo)
        controls.addWidget(self.src_combo)
        controls.addWidget(arrow)
        controls.addWidget(self.tgt_combo)
        controls.addWidget(self.toggle_btn)
        controls.addWidget(self.level, stretch=1)

        # --- hero live caption ---
        self.status = QLabel("Ready — pick a source and press Start.")
        self.status.setObjectName("status")
        self.hero = QLabel("")
        self.hero.setObjectName("hero")
        self.hero.setWordWrap(True)
        self.hero.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.heard = QLabel("")
        self.heard.setObjectName("heard")
        self.heard.setWordWrap(True)

        hero_box = QVBoxLayout()
        hero_box.setContentsMargins(24, 20, 24, 20)
        hero_box.addWidget(self.status)
        hero_box.addSpacing(8)
        hero_box.addWidget(self.hero, stretch=1)
        hero_box.addWidget(self.heard)
        hero_wrap = QWidget()
        hero_wrap.setLayout(hero_box)

        # --- history + export ---
        self.transcript = TranscriptView()
        self.export_btn = QPushButton("Export…")
        self.export_btn.setObjectName("ghost")
        self.export_btn.clicked.connect(self._on_export)
        hist_header = QHBoxLayout()
        hist_label = QLabel("History")
        hist_label.setObjectName("status")
        hist_header.addWidget(hist_label)
        hist_header.addStretch(1)
        hist_header.addWidget(self.export_btn)

        root = QVBoxLayout()
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)
        root.addLayout(controls)
        root.addWidget(hero_wrap, stretch=3)
        root.addLayout(hist_header)
        root.addWidget(self.transcript, stretch=2)
        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self.line_ready.connect(self._on_line)
        self.partial_ready.connect(self._on_partial)
        self.level_ready.connect(self.level.set_level)

    # ---- wiring from app.py ----
    def set_pipeline_factory(self, factory) -> None:
        """factory(source_kind, file_path, src, tgt) -> Pipeline."""
        self._make_pipeline = factory

    # kept for backward-compat with older app wiring / tests
    def set_pipeline(self, pipeline) -> None:
        self._pipeline = pipeline

    # ---- display ----
    def _on_partial(self, text: str) -> None:
        self.heard.setText(f"hearing:  {text}")

    def _on_line(self, line: TranslatedLine) -> None:
        self.hero.setText(line.translation)
        self.heard.setText(line.source)
        self.transcript.add_line(line.source, line.translation, line.t_start, line.t_end)

    def _on_source_changed(self, _index: int) -> None:
        if self.source_combo.currentData() == "file":
            path, _ = QFileDialog.getOpenFileName(
                self, "Choose an audio file", "", "Audio (*.wav *.aiff *.flac)")
            self._file_path = path or None
            if path:
                import os
                self.source_combo.setItemText(
                    self.source_combo.currentIndex(), f"📂  {os.path.basename(path)}")

    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export transcript", "transcript.srt", "Subtitles (*.srt);;Text (*.txt)")
        if not path:
            return
        data = self.transcript.to_srt() if path.endswith(".srt") else self.transcript.to_text()
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)

    # ---- start/stop ----
    def on_toggle(self) -> None:
        if self.toggle_btn.text() == "Start":
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        kind = self.source_combo.currentData()
        src = self.src_combo.currentData()
        tgt = self.tgt_combo.currentData()
        if kind == "file" and not self._file_path:
            self.status.setText("Pick a file first (choose “Open file…”).")
            return

        if self._make_pipeline is not None:
            self.status.setText("Loading models…")
            self._pipeline = self._make_pipeline(kind, self._file_path, src, tgt)
        if self._pipeline is None:
            return
        self.hero.setText("")
        self.heard.setText("")
        self.status.setText("Listening…  speak or play audio.")
        self.toggle_btn.setText("Stop")
        self.toggle_btn.setProperty("recording", "true")
        self.toggle_btn.style().polish(self.toggle_btn)
        self._pipeline.start()

    def _stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self.status.setText("Stopped.")
        self.toggle_btn.setText("Start")
        self.toggle_btn.setProperty("recording", "false")
        self.toggle_btn.style().polish(self.toggle_btn)
