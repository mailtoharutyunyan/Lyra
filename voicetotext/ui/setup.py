"""First-run setup: download the speech + translation models with progress.

Shown automatically when the base models are missing, so a freshly installed app
walks the user through a one-time download instead of failing on first Start.
"""
from __future__ import annotations

import shutil

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout,
)

from voicetotext import config
from voicetotext.models.download import ensure_model, missing_base_models

_QSS = """
QDialog { background: #0E1116; color: #E8EAED;
    font-family: "SF Pro Text", "Avenir Next", "Segoe UI", sans-serif; }
QLabel#title { font-family: "SF Pro Display", "Avenir Next", sans-serif;
    font-size: 22px; font-weight: 700; }
QLabel#body { color: #A9B2C2; font-size: 14px; }
QLabel#step { color: #8A93A3; font-size: 13px; }
QProgressBar { background: #1A1F29; border: none; border-radius: 6px; height: 10px; text-align: center; }
QProgressBar::chunk { background: #3DD6C4; border-radius: 6px; }
QPushButton { background: #3DD6C4; color: #08201D; border: none; border-radius: 9px;
    padding: 9px 22px; font-weight: 700; }
QPushButton:disabled { background: #26303A; color: #6B7486; }
"""


class _DownloadWorker(QThread):
    step = Signal(str)          # human status line
    advanced = Signal(int)      # models completed
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, specs) -> None:
        super().__init__()
        self._specs = specs

    def run(self) -> None:
        try:
            done = 0
            for spec in self._specs:
                mb = spec.approx_bytes // (1024 * 1024)
                self.step.emit(f"Downloading {spec.key} (~{mb} MB)…")
                ensure_model(spec)
                done += 1
                self.advanced.emit(done)
            self.finished_ok.emit()
        except Exception as e:  # surfaced in the dialog, not a crash
            self.failed.emit(str(e))


class SetupDialog(QDialog):
    """Modal first-run downloader. Returns Accepted once models are present."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set up Lyra")
        self.setStyleSheet(_QSS)
        self.setMinimumWidth(460)
        self._specs = missing_base_models()
        self._worker = None

        need = sum(s.approx_bytes for s in self._specs)
        free = shutil.disk_usage(config.models_dir()).free

        title = QLabel("Welcome to Lyra")
        title.setObjectName("title")
        body = QLabel(
            "Lyra runs entirely on your computer. It needs a one-time download of its "
            f"speech-recognition and translation models — about "
            f"{need // (1024*1024)} MB. They’re saved on this device; you won’t need to "
            "download them again."
        )
        body.setObjectName("body")
        body.setWordWrap(True)

        self.step = QLabel(f"{free // (1024*1024*1024)} GB free on disk.")
        self.step.setObjectName("step")
        self.bar = QProgressBar()
        self.bar.setRange(0, max(1, len(self._specs)))
        self.bar.setValue(0)
        self.button = QPushButton("Download and continue")
        self.button.clicked.connect(self._start)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 26, 28, 24)
        lay.setSpacing(14)
        lay.addWidget(title)
        lay.addWidget(body)
        lay.addWidget(self.step)
        lay.addWidget(self.bar)
        lay.addWidget(self.button)

    def _start(self) -> None:
        self.button.setEnabled(False)
        self._worker = _DownloadWorker(self._specs)
        self._worker.step.connect(self.step.setText)
        self._worker.advanced.connect(self.bar.setValue)
        self._worker.finished_ok.connect(self._done)
        self._worker.failed.connect(self._error)
        self._worker.start()

    def _done(self) -> None:
        self.step.setText("All set.")
        self.accept()

    def _error(self, msg: str) -> None:
        self.step.setText(f"Download failed: {msg}. Check your connection and try again.")
        self.button.setEnabled(True)
        self.button.setText("Retry")


def ensure_models_ready(parent=None) -> bool:
    """If base models are missing, run the setup dialog. Returns True when ready."""
    if not missing_base_models():
        return True
    dialog = SetupDialog(parent)
    return dialog.exec() == QDialog.Accepted
