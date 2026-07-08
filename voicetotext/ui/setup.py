"""Model download dialogs: first-run base models, and on-demand extended pack."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout

from voicetotext import config
from voicetotext.models.download import ensure_model, is_installed, missing_base_models
from voicetotext.models.registry import SEAMLESS

# Runtime components the extended (Seamless) engine needs, installed on demand.
_SEAMLESS_PACKAGES = ["torch>=2.2", "tiktoken>=0.7", "protobuf>=4.0"]


def _is_frozen() -> bool:
    """True inside a PyInstaller bundle (a downloaded app), where there is no
    Python/uv/pip to install into and the environment is read-only."""
    return getattr(sys, "frozen", False)


def _can_runtime_install() -> bool:
    """Runtime component install is only possible from source (dev) with uv/pip."""
    if _is_frozen():
        return False
    return bool(shutil.which("uv")) or importlib.util.find_spec("pip") is not None


def _install_command(packages: list[str]) -> list[str]:
    """Prefer uv (this project's manager); fall back to pip in the running venv.
    Force the running interpreter so packages land where the app can import them."""
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--python", sys.executable, *packages]
    return [sys.executable, "-m", "pip", "install", *packages]

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


def _torch_present() -> bool:
    return importlib.util.find_spec("torch") is not None


class _EnableExtendedWorker(QThread):
    step = Signal(str)
    finished_ok = Signal()
    failed = Signal(str)

    def run(self) -> None:
        try:
            if not _torch_present():
                if not _can_runtime_install():
                    # Downloaded (frozen) base edition — can't add components here.
                    self.failed.emit(
                        "Extended languages come in the Lyra Extended download. "
                        "Get it from the Lyra downloads page.")
                    return
                self.step.emit("Installing components (~250 MB)… this can take a minute.")
                result = subprocess.run(
                    _install_command(_SEAMLESS_PACKAGES),
                    capture_output=True, text=True)
                if result.returncode != 0:
                    tail = (result.stderr or "").strip().splitlines()
                    self.failed.emit(tail[-1] if tail else "component install failed")
                    return
            self.step.emit("Downloading the extended language model (~9 GB)…")
            ensure_model(SEAMLESS)
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(str(e))


class EnableExtendedDialog(QDialog):
    """One-click setup for the extended languages: installs components + model."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Extended languages")
        self.setStyleSheet(_QSS)
        self.setMinimumWidth(470)

        t = QLabel("Add extended languages"); t.setObjectName("title")
        b = QLabel(
            "This adds Armenian and 100+ other languages. Lyra will set everything up "
            "for you — it downloads about 9 GB once and keeps it on this device. You can "
            "keep using the app while it finishes.")
        b.setObjectName("body"); b.setWordWrap(True)
        self.step = QLabel("Ready when you are."); self.step.setObjectName("step")
        self.bar = QProgressBar(); self.bar.setRange(0, 0); self.bar.hide()  # busy spinner
        self.button = QPushButton("Set up extended languages")
        self.button.clicked.connect(self._start)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 26, 28, 24); lay.setSpacing(14)
        for w in (t, b, self.step, self.bar, self.button):
            lay.addWidget(w)

    def _start(self) -> None:
        self.button.setEnabled(False)
        self.bar.show()
        self._worker = _EnableExtendedWorker()
        self._worker.step.connect(self.step.setText)
        self._worker.finished_ok.connect(self.accept)
        self._worker.failed.connect(self._error)
        self._worker.start()

    def _error(self, msg: str) -> None:
        self.bar.hide()
        self.step.setText(f"Setup didn’t finish: {msg}. Please check your connection "
                          "and try again.")
        self.button.setEnabled(True)
        self.button.setText("Try again")


def ensure_seamless_ready(parent=None) -> bool:
    """Ensure the extended engine is fully usable (components + model), in-app."""
    if is_installed(SEAMLESS) and _torch_present():
        return True
    return EnableExtendedDialog(parent).exec() == QDialog.Accepted
