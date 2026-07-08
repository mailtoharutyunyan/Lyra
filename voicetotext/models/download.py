"""First-run model download with disk-space check and idempotent completion marker."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from voicetotext import config
from voicetotext.models.registry import ALL, ModelSpec


class InsufficientDiskError(Exception):
    pass


def is_installed(spec: ModelSpec) -> bool:
    return (config.models_dir() / spec.key / ".complete").exists()


def missing_base_models() -> list[ModelSpec]:
    """Base models (speech + translation) not yet downloaded."""
    return [spec for spec in ALL.values() if not is_installed(spec)]


def _default_downloader(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))


def ensure_model(
    spec: ModelSpec,
    *,
    progress: Callable[[str], None] | None = None,
    _downloader: Callable[[str, Path], None] | None = None,
) -> Path:
    local_dir = config.models_dir() / spec.key
    marker = local_dir / ".complete"
    if marker.exists():
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(local_dir).free
    if free < int(spec.approx_bytes * 1.2):
        raise InsufficientDiskError(
            f"Need ~{spec.approx_bytes // (1024*1024)} MB for {spec.key}, "
            f"only {free // (1024*1024)} MB free."
        )

    if progress:
        progress(f"Downloading {spec.key}…")
    dl = _downloader or _default_downloader
    dl(spec.repo_id, local_dir)
    marker.write_text("ok", encoding="utf-8")
    if progress:
        progress(f"{spec.key} ready.")
    return local_dir
