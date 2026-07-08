"""Application directories and persisted settings."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import platformdirs

APP_NAME = "VoiceToText"
APP_AUTHOR = "VoiceToText"

# Project root = the folder that contains the `voicetotext` package (repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Test hooks: when set, override platformdirs (see tests).
_DATA_OVERRIDE: Path | None = None
_CONFIG_OVERRIDE: Path | None = None


def data_dir() -> Path:
    base = _DATA_OVERRIDE or Path(platformdirs.user_data_dir(APP_NAME, APP_AUTHOR))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def models_dir() -> Path:
    """Where model files live.

    Priority: test override → LYRA_MODELS_DIR env → project `models/` when running
    from source (visible and manageable in the repo) → user data dir when packaged
    (an installed .app/.exe is read-only, so models must go somewhere writable).
    """
    if _DATA_OVERRIDE is not None:
        d = _DATA_OVERRIDE / "models"
    elif os.environ.get("LYRA_MODELS_DIR"):
        d = Path(os.environ["LYRA_MODELS_DIR"])
    elif not _is_frozen():
        d = PROJECT_ROOT / "models"
    else:
        d = data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_dir() -> Path:
    base = _CONFIG_OVERRIDE or Path(platformdirs.user_config_dir(APP_NAME, APP_AUTHOR))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _settings_path() -> Path:
    return config_dir() / "settings.json"


def load_settings() -> dict:
    p = _settings_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_settings(data: dict) -> None:
    _settings_path().write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
