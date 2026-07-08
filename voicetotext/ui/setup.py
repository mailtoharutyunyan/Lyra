"""Model download dialogs: first-run base models, and on-demand extended pack."""
from __future__ import annotations

import shutil

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout

from voicetotext import config
from voicetotext.models.download import ensure_model, is_installed, missing_base_models
from voicetotext.models.registry import SEAMLESS

_QSS = """
QDialog { background: #15171C; color: #E6E8EC;
    font-family: "SF Pro Text", "Segoe UI", sans-serif; }
QLabel#title { font-size: 20px; font-weight: 700; }
QLabel#body { color: #9096A0; font-size: 13px; }
QLabel#step { color: #6B7280; font-size: 12px; }
QProgressBar { background: #1D2027; border: none; border-radius: 5px; height: 8px; }
QProgressBar::chunk { background: #4C8DFF; border-radius: 5px; }
QPushButton { background: #4C8DFF; color: #06122B; border: none; border-radius: 8px;
    padding: 9px 20px; font-weight: 600; }
QPushButton:disabled { background: #262B34; color: #6B7280; }
"""


class _DownloadWorker(QThread):
    step = Signal(str)
    advanced = Signal(int)
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, specs) -> None:
        super().__init__()
        self._specs = specs

    def run(self) -> None:
        try:
            for i, spec in enumerate(self._specs, start=1):
                mb = spec.approx_bytes // (1024 * 1024)
                self.step.emit(f"Downloading {spec.key} (~{mb} MB)…")
                ensure_model(spec)
                self.advanced.emit(i)
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(str(e))


class ModelDownloadDialog(QDialog):
    def __init__(self, specs, title: str, blurb: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setStyleSheet(_QSS)
        self.setMinimumWidth(460)
        self._specs = specs

        need = sum(s.approx_bytes for s in specs)
        free = shutil.disk_usage(config.models_dir()).free

        t = QLabel(title); t.setObjectName("title")
        b = QLabel(blurb); b.setObjectName("body"); b.setWordWrap(True)
        self.step = QLabel(
            f"Need ~{need // (1024*1024)} MB · {free // (1024**3)} GB free on disk.")
        self.step.setObjectName("step")
        self.bar = QProgressBar(); self.bar.setRange(0, max(1, len(specs)))
        self.button = QPushButton("Download and continue")
        self.button.clicked.connect(self._start)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 26, 28, 24); lay.setSpacing(14)
        for w in (t, b, self.step, self.bar, self.button):
            lay.addWidget(w)

    def _start(self) -> None:
        self.button.setEnabled(False)
        self._worker = _DownloadWorker(self._specs)
        self._worker.step.connect(self.step.setText)
        self._worker.advanced.connect(self.bar.setValue)
        self._worker.finished_ok.connect(self.accept)
        self._worker.failed.connect(self._error)
        self._worker.start()

    def _error(self, msg: str) -> None:
        self.step.setText(f"Download failed: {msg}. Check your connection and retry.")
        self.button.setEnabled(True)
        self.button.setText("Retry")


def ensure_models_ready(parent=None) -> bool:
    specs = missing_base_models()
    if not specs:
        return True
    dialog = ModelDownloadDialog(
        specs, "Welcome to Lyra",
        "Lyra runs entirely on your computer. It needs a one-time download of its "
        "speech-recognition and translation models. They’re saved on this device.",
        parent)
    return dialog.exec() == QDialog.Accepted


def ensure_seamless_ready(parent=None) -> bool:
    """Ensure the extended (SeamlessM4T) model is present, downloading on demand."""
    if is_installed(SEAMLESS):
        return True
    dialog = ModelDownloadDialog(
        [SEAMLESS], "Extended language model",
        "The extended model adds Armenian and 100+ other languages. It is large "
        "(~9 GB) and downloads once, then stays on this device.",
        parent)
    return dialog.exec() == QDialog.Accepted
