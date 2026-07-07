"""Pick the right system-audio capture backend for this OS.

- macOS: macloop (CoreAudio/ScreenCaptureKit), no virtual cable needed.
- Windows: WASAPI loopback via PyAudioWPatch (no driver/permission needed).
- Fallback: BlackHole virtual device, if present.
"""
from __future__ import annotations

import sys


class SystemAudioUnavailable(Exception):
    pass


def capabilities() -> dict:
    osname = sys.platform
    if osname == "darwin":
        return {"os": osname, "system_audio": True, "method": "macloop",
                "notes": "System audio via macloop; needs Screen Recording permission."}
    if osname == "win32":
        return {"os": osname, "system_audio": True, "method": "wasapi-loopback",
                "notes": "WASAPI loopback; no setup required."}
    return {"os": osname, "system_audio": False, "method": "none",
            "notes": "System audio capture is not implemented for this OS yet."}


def make_system_source(**kw):
    caps = capabilities()
    method = caps["method"]
    if method == "macloop":
        from voicetotext.audio.macloop_source import MacloopSystemSource
        return MacloopSystemSource(**kw)
    if method == "wasapi-loopback":
        from voicetotext.audio.wasapi_loopback import WasapiLoopbackSource
        return WasapiLoopbackSource(**kw)
    raise SystemAudioUnavailable(caps["notes"])
