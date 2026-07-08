# PyInstaller spec for Lyra — cross-platform (macOS .app, Windows onedir).
# Two editions selected by env var LYRA_EDITION:
#   base (default) — no PyTorch; 25 languages + system audio; small.
#   extended       — bundles PyTorch/SeamlessM4T deps; adds Armenian + 100 languages.
# Build:  LYRA_EDITION=base pyinstaller packaging/app.spec --noconfirm
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

EDITION = os.environ.get("LYRA_EDITION", "base")
EXTENDED = EDITION == "extended"
APP_NAME = "Lyra Extended" if EXTENDED else "Lyra"
BUNDLE_ID = "com.lyra.extended" if EXTENDED else "com.lyra.app"

datas, binaries, hiddenimports = [], [], []

# Native-lib packages must be collected whole (dylibs/DLLs + data), per the
# packaging research: onnxruntime/ctranslate2/sherpa fail to load otherwise.
for pkg in ["sherpa_onnx", "sherpa_onnx_core", "ctranslate2", "soxr", "sounddevice"]:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass

# transformers is huge; we only need the NLLB tokenizer. Pull tokenizer bits.
hiddenimports += collect_submodules("sentencepiece")

_EXCLUDES = ["tensorflow", "flax", "jax",
             "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3D"]
if EXTENDED:
    # Bundle PyTorch + Seamless tokenizer deps so Extended works with no Python installed.
    for pkg in ["torch", "tiktoken"]:
        try:
            d, b, h = collect_all(pkg)
            datas += d; binaries += b; hiddenimports += h
        except Exception:
            pass
else:
    _EXCLUDES += ["torch", "torchaudio", "torchvision"]

if sys.platform == "darwin":
    try:
        d, b, h = collect_all("macloop")
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass
if sys.platform == "win32":
    try:
        d, b, h = collect_all("pyaudiowpatch")
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass

a = Analysis(
    ["launch.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=_EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name=APP_NAME, console=False, disable_windowed_traceback=False,
    argv_emulation=(sys.platform == "darwin"),
)
coll = COLLECT(exe, a.binaries, a.datas, name=APP_NAME)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        bundle_identifier=BUNDLE_ID,
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "NSMicrophoneUsageDescription":
                f"{APP_NAME} transcribes and translates microphone audio.",
            "NSAudioCaptureUsageDescription":
                f"{APP_NAME} captures system audio to caption and translate it.",
            "LSMinimumSystemVersion": "14.4",
            "NSHighResolutionCapable": True,
        },
    )
