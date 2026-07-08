"""Lyra main window: professional dark UI with source/language/model pickers,
live caption, and controls that apply without a restart."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QVBoxLayout, QWidget,
)

from voicetotext.pipeline.messages import TranslatedLine
from voicetotext.ui.languages import (
    MODEL_OPTIONS, SEAMLESS_SOURCE_CODES, SOURCE_CODES, TARGET_CODES, name_for,
)
from voicetotext.ui.transcript_view import TranscriptView
from voicetotext.ui.widgets import LevelMeter

_QSS = """
QWidget { background: #15171C; color: #E6E8EC;
    font-family: "SF Pro Text", "Segoe UI", "Inter", sans-serif; font-size: 13px; }
QLabel#wordmark { font-size: 17px; font-weight: 700; letter-spacing: 0.3px; }
QLabel#field { color: #737985; font-size: 10px; font-weight: 700;
    letter-spacing: 1.2px; }
QLabel#statusText { color: #9096A0; font-size: 12px; }
QLabel#dot { font-size: 13px; color: #6B7280; }
QLabel#dot[live="true"] { color: #34D399; }
QComboBox { background: #1D2027; border: 1px solid #2E333D; border-radius: 8px;
    padding: 7px 12px; min-height: 18px; color: #E6E8EC; }
QComboBox:hover { border-color: #3A4150; }
QComboBox:focus { border-color: #4C8DFF; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView { background: #1D2027; border: 1px solid #2E333D;
    border-radius: 8px; padding: 4px; selection-background-color: #4C8DFF;
    selection-color: #06122B; outline: none; }
QLabel#arrow { color: #4B515C; font-size: 15px; }
QPushButton#primary { background: #4C8DFF; color: #06122B; border: none;
    border-radius: 8px; padding: 9px 26px; font-weight: 600; }
QPushButton#primary:hover { background: #6BA0FF; }
QPushButton#primary[recording="true"] { background: #2E333D; color: #E6E8EC; }
QPushButton#ghost { background: transparent; color: #9096A0; border: 1px solid #2E333D;
    border-radius: 7px; padding: 6px 14px; }
QPushButton#ghost:hover { color: #E6E8EC; border-color: #3A4150; }
QFrame#card { background: #1A1D23; border: 1px solid #262B34; border-radius: 14px; }
QFrame#divider { background: #262B34; max-height: 1px; min-height: 1px; }
QLabel#capLabel { color: #737985; font-size: 10px; font-weight: 700; letter-spacing: 1.4px; }
QLabel#translation { font-family: "SF Pro Display", "Segoe UI", sans-serif;
    font-size: 26px; font-weight: 600; color: #F4F5F7; }
QLabel#heard { color: #8C93A0; font-size: 15px; }
QLabel#sectionHead { color: #737985; font-size: 10px; font-weight: 700; letter-spacing: 1.2px; }
"""


def _field(label: str, widget: QWidget) -> QWidget:
    lab = QLabel(label.upper())
    lab.setObjectName("field")
    box = QVBoxLayout()
    box.setContentsMargins(0, 0, 0, 0)
    box.setSpacing(5)
    box.addWidget(lab)
    box.addWidget(widget)
    w = QWidget()
    w.setLayout(box)
    return w


class MainWindow(QMainWindow):
    line_ready = Signal(object)
    partial_ready = Signal(str)
    level_ready = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Lyra")
        self.resize(900, 640)
        self.setStyleSheet(_QSS)
        self._make_pipeline = None
        self._pipeline = None
        self._file_path = None
        self._running = False

        # ---------- header ----------
        wordmark = QLabel("Lyra")
        wordmark.setObjectName("wordmark")
        self.dot = QLabel("●")
        self.dot.setObjectName("dot")
        self.status = QLabel("Ready")
        self.status.setObjectName("statusText")
        header = QHBoxLayout()
        header.addWidget(wordmark)
        header.addStretch(1)
        header.addWidget(self.dot)
        header.addWidget(self.status)

        # ---------- controls ----------
        self.source_combo = QComboBox()
        for label, kind in [("Microphone", "mic"), ("System audio", "system"),
                            ("Audio file…", "file")]:
            self.source_combo.addItem(label, kind)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)

        self.model_combo = QComboBox()
        for label, key in MODEL_OPTIONS:
            self.model_combo.addItem(label, key)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)

        self.src_combo = QComboBox()
        self._fill_source_langs("parakeet")
        self.src_combo.currentIndexChanged.connect(self._apply_languages)

        self.tgt_combo = QComboBox()
        for code in TARGET_CODES:
            self.tgt_combo.addItem(name_for(code), code)
        self.tgt_combo.setCurrentIndex(TARGET_CODES.index("rus_Cyrl"))
        self.tgt_combo.currentIndexChanged.connect(self._apply_languages)

        arrow = QLabel("→")
        arrow.setObjectName("arrow")
        self.toggle_btn = QPushButton("Start")
        self.toggle_btn.setObjectName("primary")
        self.toggle_btn.clicked.connect(self.on_toggle)

        controls = QHBoxLayout()
        controls.setSpacing(12)
        controls.addWidget(_field("Source", self.source_combo))
        controls.addWidget(_field("Model", self.model_combo))
        controls.addWidget(_field("From", self.src_combo))
        arrow_wrap = _field(" ", arrow)
        controls.addWidget(arrow_wrap)
        controls.addWidget(_field("To", self.tgt_combo))
        controls.addStretch(1)
        controls.addWidget(_field(" ", self.toggle_btn))

        # ---------- level meter ----------
        self.level = LevelMeter()

        # ---------- caption card ----------
        cap_label = QLabel("TRANSLATION")
        cap_label.setObjectName("capLabel")
        self.translation = QLabel("")
        self.translation.setObjectName("translation")
        self.translation.setWordWrap(True)
        self.translation.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        divider = QFrame()
        divider.setObjectName("divider")
        heard_label = QLabel("HEARD")
        heard_label.setObjectName("capLabel")
        self.heard = QLabel("")
        self.heard.setObjectName("heard")
        self.heard.setWordWrap(True)

        card_box = QVBoxLayout()
        card_box.setContentsMargins(22, 20, 22, 20)
        card_box.setSpacing(10)
        card_box.addWidget(cap_label)
        card_box.addWidget(self.translation, stretch=1)
        card_box.addWidget(divider)
        card_box.addWidget(heard_label)
        card_box.addWidget(self.heard)
        card = QFrame()
        card.setObjectName("card")
        card.setLayout(card_box)

        # ---------- history ----------
        hist_head = QLabel("HISTORY")
        hist_head.setObjectName("sectionHead")
        self.export_btn = QPushButton("Export…")
        self.export_btn.setObjectName("ghost")
        self.export_btn.clicked.connect(self._on_export)
        hist_row = QHBoxLayout()
        hist_row.addWidget(hist_head)
        hist_row.addStretch(1)
        hist_row.addWidget(self.export_btn)
        self.transcript = TranscriptView()

        # ---------- assemble ----------
        root = QVBoxLayout()
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(14)
        root.addLayout(header)
        root.addLayout(controls)
        root.addWidget(self.level)
        root.addWidget(card, stretch=3)
        root.addLayout(hist_row)
        root.addWidget(self.transcript, stretch=2)
        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self.line_ready.connect(self._on_line)
        self.partial_ready.connect(self._on_partial)
        self.level_ready.connect(self.level.set_level)

    # ---------- wiring ----------
    def set_pipeline_factory(self, factory) -> None:
        """factory(source_kind, file_path, src, tgt, model) -> Pipeline."""
        self._make_pipeline = factory

    def set_pipeline(self, pipeline) -> None:  # back-compat
        self._pipeline = pipeline

    def _fill_source_langs(self, model_key: str) -> None:
        codes = SEAMLESS_SOURCE_CODES if model_key == "seamless" else SOURCE_CODES
        self.src_combo.blockSignals(True)
        self.src_combo.clear()
        for code in codes:
            self.src_combo.addItem(name_for(code), code)
        self.src_combo.blockSignals(False)

    # ---------- display ----------
    def _set_live(self, live: bool, text: str) -> None:
        self.dot.setProperty("live", "true" if live else "false")
        self.dot.style().polish(self.dot)
        self.status.setText(text)

    def _on_partial(self, text: str) -> None:
        self.heard.setText(text)

    def _on_line(self, line: TranslatedLine) -> None:
        self.translation.setText(line.translation)
        self.heard.setText(line.source)
        self.transcript.add_line(line.source, line.translation, line.t_start, line.t_end)

    # ---------- control changes ----------
    def _apply_languages(self, *_args) -> None:
        # Live: update the running pipeline's languages without a restart.
        if self._pipeline is not None and self._running:
            self._pipeline.set_languages(self._src_code(), self.tgt_combo.currentData())

    def _src_code(self) -> str:
        code = self.src_combo.currentData()
        return "eng_Latn" if code in (None, "auto") else code

    def _on_model_changed(self, *_args) -> None:
        self._fill_source_langs(self.model_combo.currentData())
        if self._running:
            self._restart()

    def _on_source_changed(self, *_args) -> None:
        if self.source_combo.currentData() == "file":
            path, _ = QFileDialog.getOpenFileName(
                self, "Choose an audio file", "", "Audio (*.wav *.aiff *.flac)")
            self._file_path = path or None
            if path:
                self.source_combo.setItemText(
                    self.source_combo.currentIndex(), os.path.basename(path))
        if self._running:
            self._restart()

    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export transcript", "transcript.srt", "Subtitles (*.srt);;Text (*.txt)")
        if not path:
            return
        data = self.transcript.to_srt() if path.endswith(".srt") else self.transcript.to_text()
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)

    # ---------- start / stop ----------
    def on_toggle(self) -> None:
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        kind = self.source_combo.currentData()
        model = self.model_combo.currentData()
        if kind == "file" and not self._file_path:
            self._set_live(False, "Choose an audio file first.")
            return
        if model == "seamless" and not self._ensure_seamless():
            return
        if self._make_pipeline is None:
            return

        self._set_live(False, "Loading model…")
        self.toggle_btn.setEnabled(False)
        self.repaint()
        try:
            self._pipeline = self._make_pipeline(
                kind, self._file_path, self._src_code(), self.tgt_combo.currentData(), model)
        except Exception as e:
            self._set_live(False, f"Couldn’t start: {e}")
            self.toggle_btn.setEnabled(True)
            return
        self.toggle_btn.setEnabled(True)

        self.translation.setText("")
        self.heard.setText("")
        self._running = True
        self._set_live(True, "Listening — speak or play audio")
        self.toggle_btn.setText("Stop")
        self.toggle_btn.setProperty("recording", "true")
        self.toggle_btn.style().polish(self.toggle_btn)
        self._pipeline.start()

    def _stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._running = False
        self._set_live(False, "Stopped")
        self.toggle_btn.setText("Start")
        self.toggle_btn.setProperty("recording", "false")
        self.toggle_btn.style().polish(self.toggle_btn)

    def _restart(self) -> None:
        self._stop()
        self._start()

    def _ensure_seamless(self) -> bool:
        from voicetotext.asr.seamless import seamless_status
        from voicetotext.ui.setup import ensure_seamless_ready

        st = seamless_status()
        if not st["has_torch"]:
            self._set_live(False, "Extended model needs the optional pack "
                                  "(install: uv sync --extra seamless).")
            return False
        if not st["available"]:
            self._set_live(False, st["notes"])
            return False
        return ensure_seamless_ready(self)
