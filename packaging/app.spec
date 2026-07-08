# PyInstaller spec for VoiceToText — cross-platform (macOS .app, Windows onedir).
# Build:  pyinstaller packaging/app.spec --noconfirm
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

APP_NAME = "Lyra"            # product/display name (installer, .app, .exe)
BUNDLE_ID = "com.lyra.app"   # macOS bundle identifier (keep stable across releases)

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
    # Keep torch out of the base app — the Seamless pack installs it separately.
    excludes=["torch", "torchaudio", "torchvision", "tensorflow", "flax", "jax",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3D"],
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
