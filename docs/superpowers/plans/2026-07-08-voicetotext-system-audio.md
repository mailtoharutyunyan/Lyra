# VoiceToText System Audio Capture Implementation Plan (Plan 2 / M3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let the app transcribe/translate the audio a computer is *playing* (any video, call, or app) — not just the microphone — with zero virtual-cable setup on the common path.

**Architecture:** Add new `AudioSource` implementations behind the existing `voicetotext.audio.sources.AudioSource` protocol (`start(on_audio)/stop()`, delivering 16 kHz mono float32). Windows uses WASAPI loopback via PyAudioWPatch. macOS spawns a small compiled Swift helper (Core Audio process tap, macOS 14.4+) that streams raw PCM to stdout, which Python reads and resamples. BlackHole is a documented fallback. A factory picks the right source for the OS; the UI gains a "System audio" option.

**Tech Stack:** PyAudioWPatch (Windows), Swift + CoreAudio (macOS helper), existing soxr/numpy/sounddevice, PySide6.

## Global Constraints

- Same audio contract as Plan 1: 16 kHz mono float32 numpy `(N,)` delivered to `on_audio`.
- New sources MUST implement `AudioSource` exactly (`start(self, on_audio: Callable[[np.ndarray], None]) -> None`, `stop(self) -> None`) so `Pipeline` uses them unchanged.
- No work in capture callbacks beyond copy+convert (Plan 1 rule).
- macOS system audio requires macOS 14.4+ for the Core Audio tap; the helper binary must be signed (ad-hoc is fine for local dev) or the OS won't deliver audio. `NSAudioCaptureUsageDescription` must be present when packaged (Plan 5 handles Info.plist).
- Windows loopback needs no driver/permission (WASAPI loopback is built in).
- Python 3.12, `uv run` from repo root. Commit per task on `feat/core-pipeline` (no push, no Co-Authored-By).
- **Platform reality:** the Windows source cannot be functionally tested on this macOS dev machine; its tests are import/interface/guard tests only, with functional verification deferred to a Windows/CI run. The macOS path is the one verified here.

## File Structure

```
voicetotext/audio/
  system.py             # SystemAudioSource factory (picks per-OS impl) + capabilities()
  wasapi_loopback.py    # WasapiLoopbackSource (Windows, PyAudioWPatch)
  mac_tap.py            # MacTapSource (spawns Swift helper, reads PCM pipe)
  blackhole.py          # BlackHoleSource (device by name via sounddevice) + fallback detection
helpers/mac/
  AudioTapHelper.swift  # Core Audio process tap -> raw float32 PCM to stdout
  build.sh              # swiftc build + ad-hoc codesign
tests/
  test_system_source.py # factory + interface + platform-guard tests
```

Dependency addition: `PyAudioWPatch>=0.2.12` under a Windows marker in `pyproject.toml`
(`"PyAudioWPatch>=0.2.12; sys_platform == 'win32'"`).

---

### Task 1: System-audio factory and capability probe

**Files:** Create `voicetotext/audio/system.py`; Test `tests/test_system_source.py`.

**Interfaces:**
- Produces `system.capabilities() -> dict` — `{"os": "darwin"|"win32"|..., "system_audio": bool, "method": "core-audio-tap"|"wasapi-loopback"|"blackhole"|"none", "notes": str}`.
- Produces `system.make_system_source(**kw) -> AudioSource` — returns the right impl for the OS, or raises `SystemAudioUnavailable` with a helpful message.
- `system.SystemAudioUnavailable(Exception)`.

- [ ] **Step 1: Failing test** (`tests/test_system_source.py`)

```python
import sys
import numpy as np
from voicetotext.audio.system import capabilities, make_system_source, SystemAudioUnavailable


def test_capabilities_reports_current_os():
    caps = capabilities()
    assert caps["os"] == sys.platform
    assert caps["method"] in {"core-audio-tap", "wasapi-loopback", "blackhole", "none"}
    assert isinstance(caps["system_audio"], bool)


def test_make_source_returns_audiosource_or_raises():
    try:
        src = make_system_source()
        assert hasattr(src, "start") and hasattr(src, "stop")
    except SystemAudioUnavailable as e:
        assert str(e)  # must carry guidance
```

- [ ] **Step 2:** Run `uv run pytest tests/test_system_source.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement** `voicetotext/audio/system.py`

```python
"""Pick the right system-audio capture backend for this OS."""
from __future__ import annotations

import sys


class SystemAudioUnavailable(Exception):
    pass


def capabilities() -> dict:
    osname = sys.platform
    if osname == "win32":
        return {"os": osname, "system_audio": True, "method": "wasapi-loopback",
                "notes": "WASAPI loopback; no setup required."}
    if osname == "darwin":
        import platform
        rel = tuple(int(x) for x in platform.mac_ver()[0].split(".")[:2] or [0, 0])
        if rel >= (14, 4):
            return {"os": osname, "system_audio": True, "method": "core-audio-tap",
                    "notes": "Core Audio process tap; needs signed helper + audio permission."}
        return {"os": osname, "system_audio": True, "method": "blackhole",
                "notes": "macOS < 14.4: install BlackHole and select it."}
    return {"os": osname, "system_audio": False, "method": "none",
            "notes": "System audio capture not implemented for this OS."}


def make_system_source(**kw):
    caps = capabilities()
    method = caps["method"]
    if method == "wasapi-loopback":
        from voicetotext.audio.wasapi_loopback import WasapiLoopbackSource
        return WasapiLoopbackSource(**kw)
    if method == "core-audio-tap":
        from voicetotext.audio.mac_tap import MacTapSource
        return MacTapSource(**kw)
    if method == "blackhole":
        from voicetotext.audio.blackhole import BlackHoleSource
        return BlackHoleSource(**kw)
    raise SystemAudioUnavailable(caps["notes"])
```

- [ ] **Step 4:** Run test → PASS.
- [ ] **Step 5:** Commit `feat: system-audio backend factory and capability probe`.

---

### Task 2: macOS Core Audio tap Swift helper

**Files:** Create `helpers/mac/AudioTapHelper.swift`, `helpers/mac/build.sh`.

**Interfaces:** Produces a binary `helpers/mac/AudioTapHelper` that, when run, captures the default output device's audio via a Core Audio process tap and writes **raw little-endian float32, mono, 16000 Hz** to stdout continuously until killed. Prints nothing else to stdout (diagnostics to stderr).

- [ ] **Step 1: Write `helpers/mac/AudioTapHelper.swift`** — Core Audio process tap based on Apple's `CATapDescription` + `AudioHardwareCreateProcessTap` + aggregate device (reference: AudioCap). It installs an IO callback, downmixes to mono, resamples to 16 kHz (linear is acceptable here; Python re-resamples via soxr defensively), and `fwrite`s float32 frames to stdout. Emit errors to stderr with clear messages (permission denied, unsupported OS).

> Full Swift source is ~180 lines; implementer writes it following AudioCap's tap setup. Acceptance is behavioral (Task 3 test), not a line-by-line spec. Keep the stdout stream pure PCM.

- [ ] **Step 2: Write `helpers/mac/build.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
swiftc -O AudioTapHelper.swift -o AudioTapHelper \
  -framework CoreAudio -framework AudioToolbox -framework Foundation
codesign --force --sign - --entitlements <(cat <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>com.apple.security.device.audio-input</key><true/>
</dict></plist>
EOF
) AudioTapHelper
echo "built helpers/mac/AudioTapHelper"
```

- [ ] **Step 3:** Run `bash helpers/mac/build.sh` (only on macOS 14.4+ with Xcode CLT). Expected: builds and ad-hoc signs. If `swiftc` missing, report — install with `xcode-select --install`.
- [ ] **Step 4:** Commit `feat: macOS Core Audio tap helper (Swift) + build script`.

---

### Task 3: MacTapSource (Python side)

**Files:** Create `voicetotext/audio/mac_tap.py`; extend `tests/test_system_source.py`.

**Interfaces:** `MacTapSource(helper_path=None, chunk_ms=100)` implements `AudioSource`. `start` spawns the helper via `subprocess.Popen`, reads float32 PCM from its stdout on a thread, packages into 16 kHz mono chunks, and calls `on_audio`. `stop` terminates the helper. Resolves helper path from packaged location or `helpers/mac/AudioTapHelper` in dev.

- [ ] **Step 1: Failing test** — add to `tests/test_system_source.py`:

```python
import sys
import pytest

@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_mac_tap_source_imports_and_reports_missing_helper():
    from voicetotext.audio.mac_tap import MacTapSource
    src = MacTapSource(helper_path="/nonexistent/helper")
    with pytest.raises(FileNotFoundError):
        src.start(lambda chunk: None)
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement** `mac_tap.py` — `Popen([helper], stdout=PIPE, stderr=PIPE)`; reader thread does `np.frombuffer(read, np.float32)`, buffers to `chunk_ms` worth, `to_mono_16k` defensively (helper already 16k mono → passthrough), calls `on_audio`. Raise `FileNotFoundError` if helper path absent. Log stderr lines on failure.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5 (manual functional verify, macOS):** with the helper built, run `uv run voicetotext --system --tgt rus_Cyrl` while a foreign-language video plays; grant the audio permission prompt; confirm captions appear. Record result.
- [ ] **Step 6:** Commit `feat: macOS system-audio source via tap helper`.

---

### Task 4: Windows WASAPI loopback + BlackHole fallback

**Files:** Create `voicetotext/audio/wasapi_loopback.py`, `voicetotext/audio/blackhole.py`; extend test.

**Interfaces:**
- `WasapiLoopbackSource(chunk_ms=100)` implements `AudioSource` using `pyaudiowpatch` (default WASAPI loopback device → callback → `to_mono_16k`). Import of `pyaudiowpatch` is lazy inside `start`.
- `blackhole.find_blackhole_device() -> int | None`; `BlackHoleSource(chunk_ms=100)` wraps `sounddevice` on the BlackHole device, raising `SystemAudioUnavailable` if not installed.

- [ ] **Step 1: Failing test** (platform-guarded):

```python
def test_wasapi_source_imports():
    from voicetotext.audio.wasapi_loopback import WasapiLoopbackSource
    assert hasattr(WasapiLoopbackSource, "start")

def test_blackhole_detection_runs():
    from voicetotext.audio.blackhole import find_blackhole_device
    # returns None or an int; must not raise
    res = find_blackhole_device()
    assert res is None or isinstance(res, int)
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement both. **Step 4:** Run → PASS (Windows functional test deferred to CI/Windows). **Step 5:** Commit `feat: WASAPI loopback and BlackHole fallback sources`.

---

### Task 5: Wire system audio into the UI

**Files:** Modify `voicetotext/ui/main_window.py`, `voicetotext/ui/app.py`, `voicetotext/main.py`; extend `tests/test_ui.py`.

**Interfaces:** Add a source selector ("Microphone" / "System audio") to `MainWindow`; `app.build_app` accepts `use_system=False`; `main.py` gains `--system`. On selecting system audio when unavailable, show `capabilities()["notes"]` in the status bar.

- [ ] **Step 1: Failing test** — `MainWindow` has a `source_combo` with both options; selecting "System audio" on an unsupported OS disables Start and shows the note.
- [ ] **Step 2–4:** Implement, run UI test with `QT_QPA_PLATFORM=offscreen` → PASS.
- [ ] **Step 5:** Commit `feat: system-audio option in UI and CLI`.

---

## Self-Review
- M3 requirement (hear system/video audio) covered on all three paths (WASAPI, Core Audio tap, BlackHole). ✓
- Reuses the Plan 1 `AudioSource` interface unchanged → `Pipeline` and UI need no rework beyond source selection. ✓
- Platform-honest: Windows functional verification explicitly deferred to CI/Windows; macOS is the verified path here, with a manual functional step. ✓
- Signing dependency for the macOS helper is noted; packaging (Info.plist keys, notarization) is Plan 5. ✓
