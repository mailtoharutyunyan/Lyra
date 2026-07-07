# VoiceToText Core Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the working core of VoiceToText — audio (file or microphone) → VAD → Parakeet ASR → sentence split → NLLB translation → a live transcript window — as a runnable desktop app.

**Architecture:** A single Python process. Pipeline stages (capture → ASR → translate) run on worker threads connected by `queue.Queue`, passing immutable dataclass messages. Engines sit behind `Protocol` interfaces (`AudioSource`, `ASREngine`, `Translator`) so real ML backends and test fakes are interchangeable. The UI is PySide6; the ASR/MT threads marshal results to the GUI thread via Qt signals.

**Tech Stack:** Python 3.12, uv (env + deps), PySide6 (UI), sherpa-onnx (Parakeet TDT 0.6B v3 + Silero VAD), ctranslate2 + transformers tokenizer (NLLB-200 600M int8), sounddevice + soxr (audio), huggingface_hub + platformdirs (model download/storage), pytest (tests).

## Global Constraints

- **Python 3.12** exactly — sherpa-onnx / ctranslate2 / PySide6 wheels do not yet cover 3.14; pin via `uv python pin 3.12`. (The machine has 3.14; do not use it for this project.)
- **CPU-first** — no CUDA. All inference on CPU. `num_threads=4` for ASR, `intra_threads=4` for MT (leave cores for UI); do not hardcode assumptions about core count beyond these defaults.
- **Audio contract everywhere:** 16 kHz, mono, `float32` in `[-1.0, 1.0]`, as `numpy.ndarray` shape `(N,)`. Resample once, up front, before VAD.
- **Never bundle models.** Download on first run into `platformdirs.user_data_dir("VoiceToText", "VoiceToText")/models`.
- **Parakeet is offline (non-streaming).** Do NOT use `OnlineRecognizer`. Use `OfflineRecognizer.from_transducer` + Silero VAD with the "simulated streaming" pattern (re-decode open segment for partials, decode closed segment for finals).
- **Anti-flicker policy:** translate only FINAL sentences; partials are shown untranslated and never sent to the translator. Translated lines are immutable once rendered.
- **Licenses:** Parakeet CC-BY-4.0 (ok); NLLB CC-BY-NC-4.0 (non-commercial — acceptable, personal use). Keep translation behind the `Translator` protocol so it is swappable.
- **FLORES-200 language codes** for translation: `eng_Latn`, `rus_Cyrl`, `ukr_Cyrl`, `hye_Armn`, `deu_Latn`, `fra_Latn`, `spa_Latn`, etc.
- **No work in audio callbacks** — the sounddevice callback only copies samples into a queue.
- **Commit after every task** with a `feat:`/`test:`/`chore:` prefixed message. Do NOT `git push` and do NOT add a `Co-Authored-By` footer (user commits/pushes themselves; leave commits staged-and-made but never pushed).

**Scope of THIS plan (Core Pipeline = design milestones M1+M2).** Out of scope, each its own later plan: system-audio capture (WASAPI loopback / macOS Core Audio tap), the floating subtitle overlay, `.srt`/`.txt` export polish beyond a basic action, the SeamlessM4T extended-language pack, and `.exe`/`.dmg` packaging. This plan ends with a working windowed app that transcribes+translates microphone and audio-file input.

**Reference docs:** design at `docs/superpowers/specs/2026-07-08-realtime-audio-translator-design.md`; verified API/model details at `docs/IMPLEMENTATION_SPEC.md`.

---

## File Structure

```
pyproject.toml                     # project metadata + deps (uv-managed)
.python-version                    # "3.12" (written by uv python pin)
voicetotext/
  __init__.py
  config.py                        # app dirs (platformdirs), model paths, settings load/save
  audio/
    __init__.py
    resample.py                    # to_mono_16k(): downmix + soxr resample
    sources.py                     # AudioSource protocol, FileSource, MicSource
  asr/
    __init__.py
    events.py                      # PartialTranscript, FinalTranscript dataclasses
    engine.py                      # ASREngine protocol
    parakeet.py                    # ParakeetEngine (Silero VAD + OfflineRecognizer)
  translate/
    __init__.py
    base.py                        # Translator protocol
    segmenter.py                   # split_sentences(), LANG_CODES map
    nllb.py                        # NLLBTranslator (ctranslate2)
  models/
    __init__.py
    registry.py                    # ModelSpec entries (name, repo, files, size)
    download.py                    # ensure_model(): hf snapshot_download + disk check
  pipeline/
    __init__.py
    messages.py                    # TranslatedLine dataclass; re-export ASR events
    orchestrator.py                # Pipeline: threads + queues wiring source→asr→mt
  ui/
    __init__.py
    transcript_view.py             # QWidget: scrolling caption list
    main_window.py                 # QMainWindow: controls + transcript + partial row
    app.py                         # build QApplication, wire Pipeline signals→UI
  main.py                          # CLI entry: parse --file/--mic, launch app
tests/
  __init__.py
  conftest.py                      # fixtures: sine/silence wav generators, tmp model dir
  fakes.py                         # FakeAudioSource, FakeASREngine, FakeTranslator
  test_resample.py
  test_segmenter.py
  test_download.py
  test_nllb.py                     # integration; skipped if model absent
  test_parakeet.py                 # integration; skipped if model absent
  test_orchestrator.py             # uses fakes, no ML, no audio hardware
scripts/
  bench_rtf.py                     # measure ASR real-time-factor + MT ms/sentence
```

---

### Task 1: Project scaffolding, config, and dependencies

Sets up the uv-managed Python 3.12 environment, package skeleton, and the config module every other task imports. Folds in `.gitignore` and dependency declaration.

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `voicetotext/__init__.py` (empty)
- Create: `voicetotext/config.py`
- Create: `tests/__init__.py` (empty)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `voicetotext.config.APP_NAME: str` = `"VoiceToText"`
  - `voicetotext.config.data_dir() -> pathlib.Path` — models/app data root, created on call.
  - `voicetotext.config.models_dir() -> pathlib.Path` — `data_dir() / "models"`, created on call.
  - `voicetotext.config.config_dir() -> pathlib.Path` — user config root, created on call.
  - `voicetotext.config.load_settings() -> dict` — reads `config_dir()/settings.json`, returns `{}` if absent.
  - `voicetotext.config.save_settings(data: dict) -> None` — writes `settings.json` (pretty JSON).

- [ ] **Step 1: Pin Python and init the project**

Run:
```bash
cd /Users/arayikharutyunyan/Desktop/VoiceToText
uv python pin 3.12
```
Expected: creates `.python-version` containing `3.12` (uv downloads 3.12 if missing).

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "voicetotext"
version = "0.1.0"
description = "Local real-time audio transcription and translation"
requires-python = ">=3.12,<3.13"
dependencies = [
    "sherpa-onnx>=1.10",
    "ctranslate2>=4.4",
    "transformers>=4.44",
    "sentencepiece>=0.2",
    "sounddevice>=0.4.7",
    "soxr>=0.5",
    "numpy>=1.26,<2.2",
    "huggingface-hub>=0.25",
    "platformdirs>=4.2",
    "PySide6>=6.7",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-qt>=4.4"]

[project.scripts]
voicetotext = "voicetotext.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: needs downloaded ML models (deselect with -m 'not integration')",
]
```

- [ ] **Step 3: Write `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
*.egg-info/
build/
dist/
.DS_Store
tests/fixtures/*.wav
```

- [ ] **Step 4: Create the environment**

Run:
```bash
uv sync --extra dev
```
Expected: creates `.venv/` with Python 3.12 and all deps resolved. If a wheel fails to build, stop and report — do not switch Python versions.

- [ ] **Step 5: Write the failing test**

`tests/test_config.py`:
```python
import json
from voicetotext import config


def test_dirs_are_created_and_nested(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_DATA_OVERRIDE", tmp_path / "data")
    monkeypatch.setattr(config, "_CONFIG_OVERRIDE", tmp_path / "cfg")
    assert config.data_dir().is_dir()
    assert config.models_dir() == config.data_dir() / "models"
    assert config.models_dir().is_dir()
    assert config.config_dir().is_dir()


def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_CONFIG_OVERRIDE", tmp_path / "cfg")
    assert config.load_settings() == {}
    config.save_settings({"target_lang": "rus_Cyrl", "font_size": 22})
    assert config.load_settings() == {"target_lang": "rus_Cyrl", "font_size": 22}
    # written as readable JSON
    raw = (config.config_dir() / "settings.json").read_text()
    assert json.loads(raw)["target_lang"] == "rus_Cyrl"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `AttributeError` / `ImportError` (config module has no such members yet).

- [ ] **Step 7: Write `voicetotext/config.py`**

```python
"""Application directories and persisted settings."""
from __future__ import annotations

import json
from pathlib import Path

import platformdirs

APP_NAME = "VoiceToText"
APP_AUTHOR = "VoiceToText"

# Test hooks: when set, override platformdirs (see tests).
_DATA_OVERRIDE: Path | None = None
_CONFIG_OVERRIDE: Path | None = None


def data_dir() -> Path:
    base = _DATA_OVERRIDE or Path(platformdirs.user_data_dir(APP_NAME, APP_AUTHOR))
    base.mkdir(parents=True, exist_ok=True)
    return base


def models_dir() -> Path:
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
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .gitignore .python-version voicetotext/__init__.py voicetotext/config.py tests/__init__.py tests/test_config.py
git commit -m "chore: project scaffolding, uv env, and config module"
```

---

### Task 2: Audio resampling utility

Pure-function conversion of arbitrary-rate, multi-channel audio to the 16 kHz mono float32 contract. No hardware, fully unit-testable.

**Files:**
- Create: `voicetotext/audio/__init__.py` (empty)
- Create: `voicetotext/audio/resample.py`
- Test: `tests/test_resample.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `voicetotext.audio.resample.TARGET_RATE: int` = `16000`
  - `to_mono_16k(samples: np.ndarray, src_rate: int) -> np.ndarray` — accepts shape `(N,)` or `(N, channels)`, any dtype float or int16; returns float32 `(M,)` in `[-1, 1]` at 16 kHz. If `src_rate == 16000` and already mono, returns a float32 view/copy without resampling.

- [ ] **Step 1: Write the failing test**

`tests/test_resample.py`:
```python
import numpy as np
import pytest

from voicetotext.audio.resample import TARGET_RATE, to_mono_16k


def test_target_rate_is_16k():
    assert TARGET_RATE == 16000


def test_downmix_stereo_to_mono():
    stereo = np.zeros((100, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    stereo[:, 1] = -0.5
    out = to_mono_16k(stereo, 16000)
    assert out.ndim == 1
    assert out.shape == (100,)
    assert np.allclose(out, 0.0, atol=1e-6)


def test_int16_is_scaled_to_unit_float():
    ints = np.array([32767, -32768, 0], dtype=np.int16)
    out = to_mono_16k(ints, 16000)
    assert out.dtype == np.float32
    assert out[0] == pytest.approx(1.0, abs=1e-3)
    assert out[1] == pytest.approx(-1.0, abs=1e-3)
    assert out[2] == pytest.approx(0.0)


def test_resample_changes_length_proportionally():
    # 1 second of 48 kHz mono -> ~16000 samples at 16 kHz
    src = np.zeros(48000, dtype=np.float32)
    out = to_mono_16k(src, 48000)
    assert abs(len(out) - 16000) < 50


def test_passthrough_when_already_16k_mono():
    src = np.linspace(-1, 1, 16000, dtype=np.float32)
    out = to_mono_16k(src, 16000)
    assert out.dtype == np.float32
    assert len(out) == 16000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_resample.py -v`
Expected: FAIL — `ModuleNotFoundError: voicetotext.audio.resample`.

- [ ] **Step 3: Write `voicetotext/audio/resample.py`**

```python
"""Convert arbitrary PCM audio to the pipeline contract: 16 kHz mono float32."""
from __future__ import annotations

import numpy as np
import soxr

TARGET_RATE = 16000


def _to_float32(samples: np.ndarray) -> np.ndarray:
    if samples.dtype == np.float32:
        return samples
    if samples.dtype == np.float64:
        return samples.astype(np.float32)
    if samples.dtype == np.int16:
        return (samples.astype(np.float32) / 32768.0)
    if samples.dtype == np.int32:
        return (samples.astype(np.float32) / 2147483648.0)
    # last resort: assume already float-like
    return samples.astype(np.float32)


def to_mono_16k(samples: np.ndarray, src_rate: int) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim == 2:  # (frames, channels) -> mono
        samples = samples.mean(axis=1)
    data = _to_float32(samples)
    if src_rate != TARGET_RATE:
        data = soxr.resample(data, src_rate, TARGET_RATE)
    return np.ascontiguousarray(data, dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_resample.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add voicetotext/audio/__init__.py voicetotext/audio/resample.py tests/test_resample.py
git commit -m "feat: audio resampling to 16 kHz mono float32"
```

---

### Task 3: Sentence segmenter and language-code map

Pure logic that splits finalized ASR text into translatable sentences and maps Parakeet's 2-letter language codes to FLORES-200 codes. No ML.

**Files:**
- Create: `voicetotext/translate/__init__.py` (empty)
- Create: `voicetotext/translate/segmenter.py`
- Test: `tests/test_segmenter.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `split_sentences(text: str) -> list[str]` — splits on `. ! ? …` keeping the terminator, trims whitespace, drops empties. A trailing fragment with no terminator is returned as its own sentence.
  - `LANG_CODES: dict[str, str]` — 2-letter → FLORES (e.g. `"en" -> "eng_Latn"`, `"ru" -> "rus_Cyrl"`).
  - `to_flores(code: str) -> str` — returns FLORES for a 2-letter or already-FLORES code; raises `KeyError` for unknown 2-letter input.

- [ ] **Step 1: Write the failing test**

`tests/test_segmenter.py`:
```python
import pytest

from voicetotext.translate.segmenter import LANG_CODES, split_sentences, to_flores


def test_splits_on_terminators_keeping_punctuation():
    out = split_sentences("Hello there. How are you? I am fine!")
    assert out == ["Hello there.", "How are you?", "I am fine!"]


def test_trailing_fragment_without_terminator_is_kept():
    out = split_sentences("This is unfinished")
    assert out == ["This is unfinished"]


def test_ellipsis_and_extra_whitespace():
    out = split_sentences("Wait…  really?   ")
    assert out == ["Wait…", "really?"]


def test_empty_input_yields_empty_list():
    assert split_sentences("   ") == []


def test_lang_codes_cover_required_languages():
    for two in ("en", "ru", "uk", "de", "fr", "es"):
        assert two in LANG_CODES
    assert LANG_CODES["en"] == "eng_Latn"
    assert LANG_CODES["ru"] == "rus_Cyrl"


def test_to_flores_accepts_two_letter_and_flores():
    assert to_flores("en") == "eng_Latn"
    assert to_flores("hye_Armn") == "hye_Armn"  # already FLORES, passthrough
    with pytest.raises(KeyError):
        to_flores("zz")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_segmenter.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `voicetotext/translate/segmenter.py`**

```python
"""Sentence splitting and language-code mapping (FLORES-200)."""
from __future__ import annotations

import re

# Parakeet TDT v3 2-letter output -> FLORES-200 codes used by NLLB-200.
LANG_CODES: dict[str, str] = {
    "bg": "bul_Cyrl", "hr": "hrv_Latn", "cs": "ces_Latn", "da": "dan_Latn",
    "nl": "nld_Latn", "en": "eng_Latn", "et": "est_Latn", "fi": "fin_Latn",
    "fr": "fra_Latn", "de": "deu_Latn", "el": "ell_Grek", "hu": "hun_Latn",
    "it": "ita_Latn", "lv": "lvs_Latn", "lt": "lit_Latn", "mt": "mlt_Latn",
    "pl": "pol_Latn", "pt": "por_Latn", "ro": "ron_Latn", "sk": "slk_Latn",
    "sl": "slv_Latn", "es": "spa_Latn", "sv": "swe_Latn", "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl", "hy": "hye_Armn",
}

# terminators: . ! ? … (keep the terminator with the sentence)
_SENT_RE = re.compile(r".*?[.!?…]+|.+$", re.DOTALL)


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [m.group(0).strip() for m in _SENT_RE.finditer(text)]
    return [p for p in parts if p]


def to_flores(code: str) -> str:
    if "_" in code:  # already a FLORES code
        return code
    return LANG_CODES[code]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_segmenter.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add voicetotext/translate/__init__.py voicetotext/translate/segmenter.py tests/test_segmenter.py
git commit -m "feat: sentence segmenter and FLORES-200 language map"
```

---

### Task 4: Model registry and download manager

Declares the model artifacts and provides a resumable, disk-checked first-run download. The download itself is mocked in tests (no network in CI).

**Files:**
- Create: `voicetotext/models/__init__.py` (empty)
- Create: `voicetotext/models/registry.py`
- Create: `voicetotext/models/download.py`
- Test: `tests/test_download.py`

**Interfaces:**
- Consumes: `voicetotext.config.models_dir`.
- Produces:
  - `registry.ModelSpec` dataclass: `key: str`, `repo_id: str`, `approx_bytes: int`, `kind: str` (`"hf_snapshot"`).
  - `registry.PARAKEET: ModelSpec`, `registry.SILERO_VAD: ModelSpec` (both in the sherpa-onnx asr-models repo layout), `registry.NLLB: ModelSpec` (`repo_id="JustFrederik/nllb-200-distilled-600M-ct2-int8"`).
  - `registry.ALL: dict[str, ModelSpec]`.
  - `download.InsufficientDiskError(Exception)`.
  - `download.ensure_model(spec, *, progress=None, _downloader=None) -> pathlib.Path` — returns the local dir for the model; if a sentinel `.complete` marker exists, returns immediately (idempotent). Checks free disk (`spec.approx_bytes * 1.2`) before downloading; raises `InsufficientDiskError` if short. `_downloader` is an injectable callable `(repo_id, local_dir) -> None` used by tests; defaults to `huggingface_hub.snapshot_download`.

- [ ] **Step 1: Write the failing test**

`tests/test_download.py`:
```python
import pytest

from voicetotext import config
from voicetotext.models import download, registry


@pytest.fixture
def model_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_DATA_OVERRIDE", tmp_path / "data")
    return config.models_dir()


def test_registry_has_expected_specs():
    assert registry.NLLB.repo_id == "JustFrederik/nllb-200-distilled-600M-ct2-int8"
    assert registry.PARAKEET.approx_bytes > 0
    assert set(registry.ALL) == {"parakeet", "silero_vad", "nllb"}


def test_ensure_model_invokes_downloader_and_marks_complete(model_root):
    calls = []

    def fake_dl(repo_id, local_dir):
        calls.append((repo_id, local_dir))
        # simulate a downloaded file
        (local_dir / "dummy.bin").write_bytes(b"x")

    path = download.ensure_model(registry.NLLB, _downloader=fake_dl)
    assert path.is_dir()
    assert (path / ".complete").exists()
    assert calls and calls[0][0] == registry.NLLB.repo_id


def test_ensure_model_is_idempotent(model_root):
    n = {"count": 0}

    def fake_dl(repo_id, local_dir):
        n["count"] += 1

    download.ensure_model(registry.NLLB, _downloader=fake_dl)
    download.ensure_model(registry.NLLB, _downloader=fake_dl)
    assert n["count"] == 1  # second call short-circuits on .complete


def test_insufficient_disk_raises(model_root, monkeypatch):
    huge = registry.ModelSpec(key="huge", repo_id="x/y", approx_bytes=10**18, kind="hf_snapshot")
    with pytest.raises(download.InsufficientDiskError):
        download.ensure_model(huge, _downloader=lambda *a: None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_download.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `voicetotext/models/registry.py`**

```python
"""Declarations of the ML model artifacts the app downloads."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    approx_bytes: int
    kind: str  # "hf_snapshot"


# Sizes from docs/IMPLEMENTATION_SPEC.md (verified against release/HF APIs).
PARAKEET = ModelSpec(
    key="parakeet",
    repo_id="csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    approx_bytes=640 * 1024 * 1024,
    kind="hf_snapshot",
)
SILERO_VAD = ModelSpec(
    key="silero_vad",
    repo_id="csukuangfj/sherpa-onnx-silero-vad",
    approx_bytes=2 * 1024 * 1024,
    kind="hf_snapshot",
)
NLLB = ModelSpec(
    key="nllb",
    repo_id="JustFrederik/nllb-200-distilled-600M-ct2-int8",
    approx_bytes=700 * 1024 * 1024,
    kind="hf_snapshot",
)

ALL: dict[str, ModelSpec] = {m.key: m for m in (PARAKEET, SILERO_VAD, NLLB)}
```

> **Implementer note:** the exact `repo_id`s for the sherpa-onnx Parakeet/Silero mirrors must be confirmed against Hugging Face at implementation time (the sherpa-onnx models are distributed both as GitHub release tarballs and HF mirrors). If a listed HF repo does not exist, substitute the correct mirror or switch that spec's `kind` to a tarball download; keep the `ModelSpec` interface unchanged. This is the one place a placeholder id may need correcting — verify before Task 6.

- [ ] **Step 4: Write `voicetotext/models/download.py`**

```python
"""First-run model download with disk-space check and idempotent completion marker."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from voicetotext import config
from voicetotext.models.registry import ModelSpec


class InsufficientDiskError(Exception):
    pass


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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_download.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add voicetotext/models/ tests/test_download.py
git commit -m "feat: model registry and resumable download manager"
```

---

### Task 5: Translator protocol and NLLB backend

Defines the `Translator` interface and the NLLB-200 CTranslate2 implementation. The protocol is pure; the NLLB integration test is marked `integration` and skipped when the model is absent.

**Files:**
- Create: `voicetotext/translate/base.py`
- Create: `voicetotext/translate/nllb.py`
- Test: `tests/test_nllb.py`

**Interfaces:**
- Consumes: `segmenter.to_flores`, `models.registry.NLLB`, `models.download.ensure_model`.
- Produces:
  - `base.Translator` — `typing.Protocol` with `translate(self, text: str, src: str, tgt: str) -> str` (src/tgt are FLORES codes).
  - `nllb.NLLBTranslator` implementing it. Constructor `NLLBTranslator(model_dir: str | Path, intra_threads: int = 4)`. Empty/whitespace `text` returns `""` without invoking the model.
  - `nllb.load_default() -> NLLBTranslator` — calls `ensure_model(NLLB)` then constructs from that dir.

- [ ] **Step 1: Write `voicetotext/translate/base.py`** (no test of its own; it's a Protocol)

```python
"""Translator interface — keeps the MT backend swappable (license-driven, see plan)."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Translator(Protocol):
    def translate(self, text: str, src: str, tgt: str) -> str:
        """Translate `text` from FLORES code `src` to FLORES code `tgt`."""
        ...
```

- [ ] **Step 2: Write the failing test**

`tests/test_nllb.py`:
```python
import pytest

from voicetotext.translate.base import Translator


def _model_available() -> bool:
    from voicetotext import config
    return (config.models_dir() / "nllb" / ".complete").exists()


def test_nllb_class_satisfies_protocol_without_loading():
    # import must not require the model
    from voicetotext.translate.nllb import NLLBTranslator
    assert hasattr(NLLBTranslator, "translate")


@pytest.mark.integration
@pytest.mark.skipif(not _model_available(), reason="NLLB model not downloaded")
def test_nllb_translates_english_to_russian():
    from voicetotext.translate.nllb import load_default
    tr = load_default()
    assert isinstance(tr, Translator)
    out = tr.translate("Hello, how are you?", "eng_Latn", "rus_Cyrl")
    assert isinstance(out, str) and out.strip()
    # crude sanity: Cyrillic characters present
    assert any("Ѐ" <= ch <= "ӿ" for ch in out)


@pytest.mark.integration
@pytest.mark.skipif(not _model_available(), reason="NLLB model not downloaded")
def test_nllb_empty_text_is_noop():
    from voicetotext.translate.nllb import load_default
    tr = load_default()
    assert tr.translate("   ", "eng_Latn", "rus_Cyrl") == ""
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_nllb.py -v`
Expected: FAIL on `test_nllb_class_satisfies_protocol_without_loading` — `ModuleNotFoundError` (nllb.py missing). The integration tests should SKIP.

- [ ] **Step 4: Write `voicetotext/translate/nllb.py`**

```python
"""NLLB-200 translation via CTranslate2 (int8, CPU)."""
from __future__ import annotations

from pathlib import Path

from voicetotext.models.download import ensure_model
from voicetotext.models.registry import NLLB


class NLLBTranslator:
    def __init__(self, model_dir: str | Path, intra_threads: int = 4) -> None:
        import ctranslate2
        import transformers

        self._model_dir = str(model_dir)
        self._ct = ctranslate2.Translator(
            self._model_dir, device="cpu", inter_threads=1, intra_threads=intra_threads
        )
        self._tok = transformers.AutoTokenizer.from_pretrained(self._model_dir)

    def translate(self, text: str, src: str, tgt: str) -> str:
        text = text.strip()
        if not text:
            return ""
        self._tok.src_lang = src
        tokens = self._tok.convert_ids_to_tokens(self._tok.encode(text))
        results = self._ct.translate_batch([tokens], target_prefix=[[tgt]])
        out = results[0].hypotheses[0]
        if out and out[0] == tgt:  # strip target-language prefix token
            out = out[1:]
        return self._tok.decode(self._tok.convert_tokens_to_ids(out))


def load_default() -> "NLLBTranslator":
    model_dir = ensure_model(NLLB)
    return NLLBTranslator(model_dir)
```

- [ ] **Step 5: Run test to verify it passes (protocol test) / skips (integration)**

Run: `uv run pytest tests/test_nllb.py -v`
Expected: `test_nllb_class_satisfies_protocol_without_loading` PASS; two integration tests SKIP (model not yet downloaded). This is correct.

- [ ] **Step 6: Commit**

```bash
git add voicetotext/translate/base.py voicetotext/translate/nllb.py tests/test_nllb.py
git commit -m "feat: Translator protocol and NLLB-200 CTranslate2 backend"
```

---

### Task 6: ASR events, engine protocol, and Parakeet backend

Defines the transcript event types, the `ASREngine` protocol, and the Parakeet implementation using Silero VAD + `OfflineRecognizer` with the simulated-streaming pattern.

**Files:**
- Create: `voicetotext/asr/__init__.py` (empty)
- Create: `voicetotext/asr/events.py`
- Create: `voicetotext/asr/engine.py`
- Create: `voicetotext/asr/parakeet.py`
- Test: `tests/test_parakeet.py`

**Interfaces:**
- Consumes: `audio.resample.TARGET_RATE`, `models.registry.PARAKEET/SILERO_VAD`, `models.download.ensure_model`.
- Produces:
  - `events.PartialTranscript` (frozen dataclass): `text: str`.
  - `events.FinalTranscript` (frozen dataclass): `text: str`, `t_start: float`, `t_end: float`.
  - `events.Event = PartialTranscript | FinalTranscript`.
  - `engine.ASREngine` Protocol: `accept(self, samples: np.ndarray) -> list[Event]` (samples are 16 kHz mono float32) and `flush(self) -> list[Event]` (finalize any open segment, e.g. at stream end).
  - `parakeet.ParakeetEngine` implementing it. Constructor `ParakeetEngine(model_dir, vad_model_path, *, num_threads=4, partial_interval_s=0.4, min_silence_s=0.15, max_speech_s=8.0)`.
  - `parakeet.load_default() -> ParakeetEngine`.

- [ ] **Step 1: Write `voicetotext/asr/events.py`**

```python
"""Transcript events emitted by an ASR engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class PartialTranscript:
    text: str


@dataclass(frozen=True)
class FinalTranscript:
    text: str
    t_start: float
    t_end: float


Event = Union[PartialTranscript, FinalTranscript]
```

- [ ] **Step 2: Write `voicetotext/asr/engine.py`**

```python
"""ASR engine interface."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from voicetotext.asr.events import Event


class ASREngine(Protocol):
    def accept(self, samples: np.ndarray) -> list[Event]:
        """Feed 16 kHz mono float32 samples; return any events produced now."""
        ...

    def flush(self) -> list[Event]:
        """Finalize any open segment (e.g. at end of a file)."""
        ...
```

- [ ] **Step 3: Write the failing test**

`tests/test_parakeet.py`:
```python
import numpy as np
import pytest

from voicetotext.asr.events import FinalTranscript, PartialTranscript


def _models_available() -> bool:
    from voicetotext import config
    return (
        (config.models_dir() / "parakeet" / ".complete").exists()
        and (config.models_dir() / "silero_vad" / ".complete").exists()
    )


def test_event_types_are_frozen():
    p = PartialTranscript(text="hi")
    f = FinalTranscript(text="hi.", t_start=0.0, t_end=1.0)
    with pytest.raises(Exception):
        p.text = "x"  # frozen
    assert f.t_end == 1.0


def test_engine_import_does_not_require_models():
    from voicetotext.asr.parakeet import ParakeetEngine
    assert hasattr(ParakeetEngine, "accept")
    assert hasattr(ParakeetEngine, "flush")


@pytest.mark.integration
@pytest.mark.skipif(not _models_available(), reason="Parakeet/VAD models not downloaded")
def test_silence_produces_no_finals():
    from voicetotext.asr.parakeet import load_default
    eng = load_default()
    silence = np.zeros(16000, dtype=np.float32)  # 1 s
    events = eng.accept(silence) + eng.flush()
    assert not any(isinstance(e, FinalTranscript) for e in events)


@pytest.mark.integration
@pytest.mark.skipif(not _models_available(), reason="Parakeet/VAD models not downloaded")
def test_speech_fixture_produces_a_final(speech_wav_16k):
    # speech_wav_16k fixture: (samples float32 16k, expected_substring)
    from voicetotext.asr.parakeet import load_default
    samples, expected = speech_wav_16k
    eng = load_default()
    events = []
    # feed in 100 ms chunks to exercise the streaming path
    for i in range(0, len(samples), 1600):
        events += eng.accept(samples[i : i + 1600])
    events += eng.flush()
    finals = [e for e in events if isinstance(e, FinalTranscript)]
    assert finals, "expected at least one final transcript"
    joined = " ".join(e.text.lower() for e in finals)
    assert expected.lower() in joined
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_parakeet.py -v`
Expected: `test_event_types_are_frozen` and `test_engine_import_does_not_require_models` FAIL first (`ModuleNotFoundError`); integration tests SKIP. After Step 5 the first two PASS.

- [ ] **Step 5: Write `voicetotext/asr/parakeet.py`**

```python
"""Parakeet TDT 0.6B v3 offline ASR with Silero VAD (simulated streaming)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from voicetotext.asr.events import Event, FinalTranscript, PartialTranscript
from voicetotext.audio.resample import TARGET_RATE
from voicetotext.models.download import ensure_model
from voicetotext.models.registry import PARAKEET, SILERO_VAD

_VAD_WINDOW = 512  # samples @ 16 kHz (Silero requirement)


class ParakeetEngine:
    def __init__(
        self,
        model_dir: str | Path,
        vad_model_path: str | Path,
        *,
        num_threads: int = 4,
        partial_interval_s: float = 0.4,
        min_silence_s: float = 0.15,
        max_speech_s: float = 8.0,
    ) -> None:
        import sherpa_onnx

        model_dir = Path(model_dir)
        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(next(model_dir.glob("encoder*.onnx"))),
            decoder=str(next(model_dir.glob("decoder*.onnx"))),
            joiner=str(next(model_dir.glob("joiner*.onnx"))),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=num_threads,
            model_type="nemo_transducer",
        )
        vad_cfg = sherpa_onnx.VadModelConfig()
        vad_cfg.silero_vad.model = str(vad_model_path)
        vad_cfg.silero_vad.threshold = 0.5
        vad_cfg.silero_vad.min_silence_duration = min_silence_s
        vad_cfg.silero_vad.min_speech_duration = 0.25
        vad_cfg.silero_vad.max_speech_duration = max_speech_s
        vad_cfg.sample_rate = TARGET_RATE
        self._vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=60)

        self._partial_interval = int(partial_interval_s * TARGET_RATE)
        self._buf = np.empty(0, dtype=np.float32)          # feeds VAD in 512-hops
        self._open_segment = np.empty(0, dtype=np.float32)  # accumulates current speech
        self._samples_since_partial = 0
        self._t = 0.0  # seconds consumed, for timestamps

    def _decode(self, samples: np.ndarray) -> str:
        stream = self._rec.create_stream()
        stream.accept_waveform(TARGET_RATE, samples)
        self._rec.decode_stream(stream)
        return stream.result.text.strip()

    def accept(self, samples: np.ndarray) -> list[Event]:
        events: list[Event] = []
        self._buf = np.concatenate([self._buf, samples.astype(np.float32)])

        while len(self._buf) >= _VAD_WINDOW:
            window = self._buf[:_VAD_WINDOW]
            self._buf = self._buf[_VAD_WINDOW:]
            self._t += _VAD_WINDOW / TARGET_RATE
            self._vad.accept_waveform(window)

            if self._vad.is_speech_detected():
                self._open_segment = np.concatenate([self._open_segment, window])
                self._samples_since_partial += _VAD_WINDOW
                if self._samples_since_partial >= self._partial_interval:
                    self._samples_since_partial = 0
                    text = self._decode(self._open_segment)
                    if text:
                        events.append(PartialTranscript(text=text))

            # drain any segments the VAD has closed
            while not self._vad.empty():
                seg = self._vad.front.samples
                self._vad.pop()
                text = self._decode(np.asarray(seg, dtype=np.float32))
                seg_dur = len(seg) / TARGET_RATE
                if text:
                    events.append(
                        FinalTranscript(text=text, t_start=self._t - seg_dur, t_end=self._t)
                    )
                self._open_segment = np.empty(0, dtype=np.float32)
                self._samples_since_partial = 0

        return events

    def flush(self) -> list[Event]:
        events: list[Event] = []
        self._vad.flush()
        while not self._vad.empty():
            seg = self._vad.front.samples
            self._vad.pop()
            text = self._decode(np.asarray(seg, dtype=np.float32))
            if text:
                seg_dur = len(seg) / TARGET_RATE
                events.append(
                    FinalTranscript(text=text, t_start=self._t - seg_dur, t_end=self._t)
                )
        self._open_segment = np.empty(0, dtype=np.float32)
        return events


def load_default() -> "ParakeetEngine":
    model_dir = ensure_model(PARAKEET)
    vad_dir = ensure_model(SILERO_VAD)
    vad_path = next(Path(vad_dir).glob("*silero_vad*.onnx"))
    return ParakeetEngine(model_dir, vad_path)
```

> **Implementer note:** the exact sherpa-onnx VAD API surface (`vad.front.samples`, `is_speech_detected()`, `flush()`) should be confirmed against the installed `sherpa-onnx` version's `python-api-examples/vad-with-non-streaming-asr.py`. If attribute names differ, adjust to match the installed version — the event-emission logic and interface stay the same. Verify with `uv run python -c "import sherpa_onnx; help(sherpa_onnx.VoiceActivityDetector)"`.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_parakeet.py -v`
Expected: 2 PASS, integration tests SKIP.

- [ ] **Step 7: Commit**

```bash
git add voicetotext/asr/ tests/test_parakeet.py
git commit -m "feat: Parakeet ASR engine with Silero VAD simulated streaming"
```

---

### Task 7: Audio sources (file + microphone)

The `AudioSource` interface plus a `FileSource` (drives the pipeline from a WAV — the CI/dev workhorse) and a `MicSource` (sounddevice). FileSource is fully unit-tested; MicSource has an import/smoke test only.

**Files:**
- Create: `voicetotext/audio/sources.py`
- Modify: `tests/conftest.py` (add WAV fixtures — created in Task 9's conftest if not present; see note)
- Test: `tests/test_sources.py`

**Interfaces:**
- Consumes: `audio.resample.to_mono_16k`, `audio.resample.TARGET_RATE`.
- Produces:
  - `sources.AudioSource` Protocol: `start(self, on_audio: Callable[[np.ndarray], None]) -> None`, `stop(self) -> None`. `on_audio` receives 16 kHz mono float32 chunks.
  - `sources.FileSource(path, chunk_ms=100, realtime=False)` — reads a WAV, resamples, and delivers it in `chunk_ms` chunks. `realtime=False` (test mode) delivers as fast as possible and returns when done; `realtime=True` paces with `time.sleep` for demo. `start` is blocking in file mode (caller runs it on a thread).
  - `sources.MicSource(device=None, blocksize_ms=100)` — opens `sounddevice.InputStream`; callback resamples and forwards. `start` is non-blocking.
  - `sources.RMS(samples) -> float` helper for the level meter.

- [ ] **Step 1: Write the failing test**

`tests/test_sources.py`:
```python
import wave

import numpy as np

from voicetotext.audio.sources import FileSource, RMS


def _write_wav(path, samples_i16, rate):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples_i16.tobytes())


def test_rms_of_silence_is_zero_and_signal_positive():
    assert RMS(np.zeros(100, dtype=np.float32)) == 0.0
    assert RMS(np.full(100, 0.5, dtype=np.float32)) > 0.4


def test_file_source_delivers_all_audio_as_16k_mono(tmp_path):
    rate = 48000
    samples = (np.sin(np.linspace(0, 100, rate)) * 10000).astype(np.int16)  # 1 s
    wav = tmp_path / "t.wav"
    _write_wav(wav, samples, rate)

    got = []
    src = FileSource(wav, chunk_ms=100, realtime=False)
    src.start(lambda chunk: got.append(chunk))

    total = np.concatenate(got)
    assert total.dtype == np.float32
    assert abs(len(total) - 16000) < 200      # ~1 s resampled to 16 kHz
    assert all(c.ndim == 1 for c in got)


def test_mic_source_imports():
    from voicetotext.audio.sources import MicSource
    assert hasattr(MicSource, "start") and hasattr(MicSource, "stop")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sources.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `voicetotext/audio/sources.py`**

```python
"""Audio sources: file playback (dev/test) and live microphone."""
from __future__ import annotations

import time
import wave
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np

from voicetotext.audio.resample import TARGET_RATE, to_mono_16k

OnAudio = Callable[[np.ndarray], None]


def RMS(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


class AudioSource(Protocol):
    def start(self, on_audio: OnAudio) -> None: ...
    def stop(self) -> None: ...


class FileSource:
    def __init__(self, path: str | Path, chunk_ms: int = 100, realtime: bool = False) -> None:
        self._path = Path(path)
        self._chunk_ms = chunk_ms
        self._realtime = realtime
        self._stopped = False

    def start(self, on_audio: OnAudio) -> None:
        self._stopped = False
        with wave.open(str(self._path), "rb") as w:
            rate = w.getframerate()
            channels = w.getnchannels()
            width = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[width]
        data = np.frombuffer(raw, dtype=dtype)
        if channels > 1:
            data = data.reshape(-1, channels)
        mono16k = to_mono_16k(data, rate)

        step = int(TARGET_RATE * self._chunk_ms / 1000)
        for i in range(0, len(mono16k), step):
            if self._stopped:
                break
            chunk = mono16k[i : i + step]
            on_audio(chunk)
            if self._realtime:
                time.sleep(self._chunk_ms / 1000)

    def stop(self) -> None:
        self._stopped = True


class MicSource:
    def __init__(self, device: Optional[int] = None, blocksize_ms: int = 100) -> None:
        self._device = device
        self._blocksize_ms = blocksize_ms
        self._stream = None

    def start(self, on_audio: OnAudio) -> None:
        import sounddevice as sd

        info = sd.query_devices(self._device, "input")
        native_rate = int(info["default_samplerate"])
        blocksize = int(native_rate * self._blocksize_ms / 1000)

        def _cb(indata, frames, time_info, status):  # runs on PortAudio thread
            chunk = to_mono_16k(indata.copy(), native_rate)  # copy: no work beyond convert
            on_audio(chunk)

        self._stream = sd.InputStream(
            samplerate=native_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            device=self._device,
            callback=_cb,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sources.py -v`
Expected: PASS (3 passed). `test_mic_source_imports` passes without opening a device.

- [ ] **Step 5: Commit**

```bash
git add voicetotext/audio/sources.py tests/test_sources.py
git commit -m "feat: file and microphone audio sources"
```

---

### Task 8: Pipeline messages and orchestrator

Wires source → ASR → translator across threads with queues, applying the anti-flicker policy (translate finals only). Tested end-to-end with fakes — no ML, no audio hardware.

**Files:**
- Create: `voicetotext/pipeline/__init__.py` (empty)
- Create: `voicetotext/pipeline/messages.py`
- Create: `voicetotext/pipeline/orchestrator.py`
- Create: `tests/fakes.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `asr.engine.ASREngine`, `asr.events.*`, `translate.base.Translator`, `translate.segmenter.split_sentences`, `audio.sources.AudioSource`.
- Produces:
  - `messages.TranslatedLine` (frozen dataclass): `source: str`, `translation: str`, `t_start: float`, `t_end: float`.
  - `messages.Partial` re-exported alias of `PartialTranscript`.
  - `orchestrator.Pipeline(source, engine, translator, src_lang, tgt_lang, on_partial, on_line)`:
    - `on_partial: Callable[[str], None]` — called with untranslated partial text.
    - `on_line: Callable[[TranslatedLine], None]` — called per finalized+translated sentence.
    - `start() -> None` (spawns capture + asr + mt threads), `stop() -> None` (joins), `run_file_blocking() -> None` (for file sources/tests: run to completion synchronously).
    - `set_languages(src, tgt) -> None`.

- [ ] **Step 1: Write `tests/fakes.py`**

```python
"""Test doubles implementing the pipeline protocols (no ML, no hardware)."""
from __future__ import annotations

import numpy as np

from voicetotext.asr.events import Event, FinalTranscript, PartialTranscript


class FakeAudioSource:
    """Emits a fixed list of chunks synchronously, then returns."""
    def __init__(self, chunks: list[np.ndarray]) -> None:
        self._chunks = chunks
        self._stopped = False

    def start(self, on_audio):
        for c in self._chunks:
            if self._stopped:
                break
            on_audio(c)

    def stop(self):
        self._stopped = True


class ScriptedASREngine:
    """Returns pre-programmed events keyed by how many accept() calls have happened."""
    def __init__(self, script: dict[int, list[Event]], final_on_flush: list[Event] | None = None):
        self._script = script
        self._flush = final_on_flush or []
        self._n = 0

    def accept(self, samples) -> list[Event]:
        events = self._script.get(self._n, [])
        self._n += 1
        return list(events)

    def flush(self) -> list[Event]:
        return list(self._flush)


class RecordingTranslator:
    def __init__(self, mapping: dict[str, str] | None = None):
        self.calls: list[tuple[str, str, str]] = []
        self._map = mapping or {}

    def translate(self, text: str, src: str, tgt: str) -> str:
        self.calls.append((text, src, tgt))
        return self._map.get(text, f"[{tgt}]{text}")
```

- [ ] **Step 2: Write the failing test**

`tests/test_orchestrator.py`:
```python
import numpy as np

from voicetotext.asr.events import FinalTranscript, PartialTranscript
from voicetotext.pipeline.messages import TranslatedLine
from voicetotext.pipeline.orchestrator import Pipeline
from tests.fakes import FakeAudioSource, RecordingTranslator, ScriptedASREngine


def test_finals_are_split_and_translated_partials_are_not():
    chunks = [np.zeros(1600, dtype=np.float32) for _ in range(3)]
    source = FakeAudioSource(chunks)
    engine = ScriptedASREngine(
        script={
            0: [PartialTranscript(text="Hello")],
            1: [PartialTranscript(text="Hello there")],
            2: [FinalTranscript(text="Hello there. How are you?", t_start=0.0, t_end=2.0)],
        }
    )
    tr = RecordingTranslator()
    partials, lines = [], []

    pipe = Pipeline(
        source=source, engine=engine, translator=tr,
        src_lang="eng_Latn", tgt_lang="rus_Cyrl",
        on_partial=partials.append, on_line=lines.append,
    )
    pipe.run_file_blocking()

    # partials surfaced untranslated
    assert "Hello" in partials and "Hello there" in partials
    # the final was split into two sentences, each translated once
    assert [c[0] for c in tr.calls] == ["Hello there.", "How are you?"]
    assert all(c[1:] == ("eng_Latn", "rus_Cyrl") for c in tr.calls)
    # two immutable translated lines emitted
    assert len(lines) == 2
    assert isinstance(lines[0], TranslatedLine)
    assert lines[0].source == "Hello there."
    assert lines[0].translation == "[rus_Cyrl]Hello there."


def test_flush_final_is_translated():
    source = FakeAudioSource([np.zeros(1600, dtype=np.float32)])
    engine = ScriptedASREngine(
        script={},
        final_on_flush=[FinalTranscript(text="Bye.", t_start=0.0, t_end=1.0)],
    )
    tr = RecordingTranslator()
    lines = []
    pipe = Pipeline(
        source=source, engine=engine, translator=tr,
        src_lang="eng_Latn", tgt_lang="rus_Cyrl",
        on_partial=lambda _: None, on_line=lines.append,
    )
    pipe.run_file_blocking()
    assert [c[0] for c in tr.calls] == ["Bye."]
    assert len(lines) == 1 and lines[0].translation == "[rus_Cyrl]Bye."
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError` for messages/orchestrator.

- [ ] **Step 4: Write `voicetotext/pipeline/messages.py`**

```python
"""Messages passed between pipeline stages and to the UI."""
from __future__ import annotations

from dataclasses import dataclass

from voicetotext.asr.events import FinalTranscript, PartialTranscript  # re-export

Partial = PartialTranscript


@dataclass(frozen=True)
class TranslatedLine:
    source: str
    translation: str
    t_start: float
    t_end: float
```

- [ ] **Step 5: Write `voicetotext/pipeline/orchestrator.py`**

```python
"""Threaded orchestration: audio source -> ASR -> translation -> callbacks."""
from __future__ import annotations

import queue
import threading
from typing import Callable

import numpy as np

from voicetotext.asr.engine import ASREngine
from voicetotext.asr.events import FinalTranscript, PartialTranscript
from voicetotext.audio.sources import AudioSource
from voicetotext.pipeline.messages import TranslatedLine
from voicetotext.translate.base import Translator
from voicetotext.translate.segmenter import split_sentences

_SENTINEL = object()


class Pipeline:
    def __init__(
        self,
        source: AudioSource,
        engine: ASREngine,
        translator: Translator,
        src_lang: str,
        tgt_lang: str,
        on_partial: Callable[[str], None],
        on_line: Callable[[TranslatedLine], None],
    ) -> None:
        self._source = source
        self._engine = engine
        self._translator = translator
        self._src = src_lang
        self._tgt = tgt_lang
        self._on_partial = on_partial
        self._on_line = on_line

        self._audio_q: queue.Queue = queue.Queue(maxsize=100)
        self._final_q: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._running = False

    def set_languages(self, src: str, tgt: str) -> None:
        self._src, self._tgt = src, tgt

    # ---- stage handlers (shared by threaded and blocking modes) ----
    def _handle_audio(self, chunk: np.ndarray) -> None:
        for ev in self._engine.accept(chunk):
            self._dispatch_event(ev)

    def _dispatch_event(self, ev) -> None:
        if isinstance(ev, PartialTranscript):
            self._on_partial(ev.text)
        elif isinstance(ev, FinalTranscript):
            self._translate_final(ev)

    def _translate_final(self, ev: FinalTranscript) -> None:
        for sentence in split_sentences(ev.text):
            translation = self._translator.translate(sentence, self._src, self._tgt)
            self._on_line(
                TranslatedLine(
                    source=sentence,
                    translation=translation,
                    t_start=ev.t_start,
                    t_end=ev.t_end,
                )
            )

    # ---- blocking mode (file sources, tests) ----
    def run_file_blocking(self) -> None:
        self._source.start(self._handle_audio)
        for ev in self._engine.flush():
            self._dispatch_event(ev)

    # ---- threaded mode (live mic) ----
    def start(self) -> None:
        self._running = True

        def capture():
            self._source.start(lambda chunk: self._audio_q.put(chunk))
            self._audio_q.put(_SENTINEL)

        def asr():
            while True:
                item = self._audio_q.get()
                if item is _SENTINEL:
                    for ev in self._engine.flush():
                        self._final_q.put(ev)
                    self._final_q.put(_SENTINEL)
                    return
                for ev in self._engine.accept(item):
                    if isinstance(ev, PartialTranscript):
                        self._on_partial(ev.text)  # partial: straight to UI
                    else:
                        self._final_q.put(ev)

        def mt():
            while True:
                ev = self._final_q.get()
                if ev is _SENTINEL:
                    return
                self._translate_final(ev)

        self._threads = [
            threading.Thread(target=capture, daemon=True, name="capture"),
            threading.Thread(target=asr, daemon=True, name="asr"),
            threading.Thread(target=mt, daemon=True, name="mt"),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._running = False
        self._source.stop()
        for t in self._threads:
            t.join(timeout=5)
        self._threads = []
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Run the whole non-integration suite**

Run: `uv run pytest -m "not integration" -v`
Expected: all tests PASS (integration tests skipped).

- [ ] **Step 8: Commit**

```bash
git add voicetotext/pipeline/ tests/fakes.py tests/test_orchestrator.py
git commit -m "feat: threaded pipeline orchestrator with anti-flicker policy"
```

---

### Task 9: Transcript UI and application wiring

The PySide6 transcript view + main window, and `main.py` wiring a real pipeline. Pipeline callbacks (which fire on worker threads) marshal to the GUI thread via Qt signals. Includes a `conftest.py` for the WAV fixtures referenced earlier.

**Files:**
- Create: `voicetotext/ui/__init__.py` (empty)
- Create: `voicetotext/ui/transcript_view.py`
- Create: `voicetotext/ui/main_window.py`
- Create: `voicetotext/ui/app.py`
- Create: `voicetotext/main.py`
- Create: `tests/conftest.py`
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `transcript_view.TranscriptView(QWidget)` — `add_line(source: str, translation: str)`, `set_partial(text: str)`, `clear()`, `to_srt() -> str`, `to_text() -> str`.
  - `main_window.MainWindow(QMainWindow)` — holds a `TranscriptView`, source/target language combo boxes, a start/stop button, and a level meter (`QProgressBar`). Signals: `line_ready(object)`, `partial_ready(str)`, `level_ready(float)` (thread-safe bridges). Slot `on_toggle()` starts/stops the pipeline.
  - `app.build_app(argv, *, file_path=None, use_mic=False) -> tuple[QApplication, MainWindow]`.
  - `main.main() -> int` — argparse `--file PATH` / `--mic` / `--src` / `--tgt`, launches the app.

- [ ] **Step 1: Write `tests/conftest.py`**

```python
import numpy as np
import pytest


@pytest.fixture
def speech_wav_16k():
    """Placeholder speech fixture.

    Replace the .npy load below with a real 16 kHz mono float32 recording of a
    known phrase for the integration test to assert on. Until then this fixture
    returns silence and the dependent integration test will simply not match.
    """
    from pathlib import Path
    p = Path(__file__).parent / "fixtures" / "speech_16k.npy"
    if p.exists():
        samples = np.load(p).astype(np.float32)
        return samples, "hello"  # expected substring for the recorded phrase
    return np.zeros(16000, dtype=np.float32), "hello"
```

- [ ] **Step 2: Write the failing UI test**

`tests/test_ui.py` (uses `pytest-qt`'s `qtbot`):
```python
import pytest

pytest.importorskip("PySide6")


def test_transcript_view_add_and_export(qtbot):
    from voicetotext.ui.transcript_view import TranscriptView
    v = TranscriptView()
    qtbot.addWidget(v)
    v.add_line("Hello there.", "Привет.")
    v.add_line("How are you?", "Как дела?")
    text = v.to_text()
    assert "Hello there." in text and "Привет." in text
    srt = v.to_srt()
    assert "1\n" in srt and "-->" in srt  # SRT numbering + timing arrow


def test_set_partial_is_replaced_not_appended(qtbot):
    from voicetotext.ui.transcript_view import TranscriptView
    v = TranscriptView()
    qtbot.addWidget(v)
    v.set_partial("Hel")
    v.set_partial("Hello")
    assert v.current_partial() == "Hello"


def test_main_window_builds(qtbot):
    from voicetotext.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.windowTitle() == "VoiceToText"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_ui.py -v`
Expected: FAIL — `ModuleNotFoundError` for the ui modules.

- [ ] **Step 4: Write `voicetotext/ui/transcript_view.py`**

```python
"""Scrolling transcript with export to plain text and SRT."""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget, QLabel


@dataclass
class _Line:
    source: str
    translation: str
    t_start: float = 0.0
    t_end: float = 0.0


def _srt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class TranscriptView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._lines: list[_Line] = []
        self._partial = ""
        self._list = QListWidget()
        self._partial_label = QLabel("")
        self._partial_label.setStyleSheet("color: gray; font-style: italic;")
        layout = QVBoxLayout(self)
        layout.addWidget(self._list, stretch=1)
        layout.addWidget(self._partial_label)

    def add_line(self, source: str, translation: str, t_start: float = 0.0, t_end: float = 0.0) -> None:
        self._lines.append(_Line(source, translation, t_start, t_end))
        item = QListWidgetItem(f"{source}\n    → {translation}")
        self._list.addItem(item)
        self._list.scrollToBottom()
        self.set_partial("")  # a finalized line clears the live row

    def set_partial(self, text: str) -> None:
        self._partial = text
        self._partial_label.setText(text)

    def current_partial(self) -> str:
        return self._partial

    def clear(self) -> None:
        self._lines.clear()
        self._list.clear()
        self.set_partial("")

    def to_text(self) -> str:
        return "\n".join(f"{ln.source}\n{ln.translation}" for ln in self._lines)

    def to_srt(self) -> str:
        blocks = []
        for i, ln in enumerate(self._lines, start=1):
            blocks.append(
                f"{i}\n{_srt_ts(ln.t_start)} --> {_srt_ts(ln.t_end)}\n"
                f"{ln.source}\n{ln.translation}\n"
            )
        return "\n".join(blocks)
```

- [ ] **Step 5: Write `voicetotext/ui/main_window.py`**

```python
"""Main application window: controls, transcript, level meter."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QMainWindow, QProgressBar, QPushButton,
    QVBoxLayout, QWidget,
)

from voicetotext.pipeline.messages import TranslatedLine
from voicetotext.translate.segmenter import LANG_CODES
from voicetotext.ui.transcript_view import TranscriptView

_TARGETS = ["eng_Latn", "rus_Cyrl", "ukr_Cyrl", "hye_Armn", "deu_Latn", "fra_Latn", "spa_Latn"]


class MainWindow(QMainWindow):
    # thread-safe bridges: pipeline worker threads emit these; slots run on GUI thread
    line_ready = Signal(object)
    partial_ready = Signal(str)
    level_ready = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VoiceToText")
        self.resize(700, 500)
        self._pipeline = None  # set by app wiring

        self.src_combo = QComboBox()
        self.src_combo.addItem("auto", "auto")
        for two, flores in LANG_CODES.items():
            self.src_combo.addItem(f"{two} ({flores})", flores)
        self.tgt_combo = QComboBox()
        for flores in _TARGETS:
            self.tgt_combo.addItem(flores, flores)
        self.tgt_combo.setCurrentText("rus_Cyrl")

        self.toggle_btn = QPushButton("Start")
        self.toggle_btn.clicked.connect(self.on_toggle)
        self.level = QProgressBar()
        self.level.setRange(0, 100)
        self.level.setTextVisible(False)

        controls = QHBoxLayout()
        controls.addWidget(self.src_combo)
        controls.addWidget(self.tgt_combo)
        controls.addWidget(self.toggle_btn)
        controls.addWidget(self.level, stretch=1)

        self.transcript = TranscriptView()

        root = QVBoxLayout()
        root.addLayout(controls)
        root.addWidget(self.transcript, stretch=1)
        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self.line_ready.connect(self._on_line)
        self.partial_ready.connect(self.transcript.set_partial)
        self.level_ready.connect(lambda v: self.level.setValue(int(min(1.0, v * 3) * 100)))

    def set_pipeline(self, pipeline) -> None:
        self._pipeline = pipeline

    def _on_line(self, line: TranslatedLine) -> None:
        self.transcript.add_line(line.source, line.translation, line.t_start, line.t_end)

    def on_toggle(self) -> None:
        if self._pipeline is None:
            return
        if self.toggle_btn.text() == "Start":
            self._pipeline.set_languages(self.src_combo.currentData(), self.tgt_combo.currentData())
            self._pipeline.start()
            self.toggle_btn.setText("Stop")
        else:
            self._pipeline.stop()
            self.toggle_btn.setText("Start")
```

- [ ] **Step 6: Write `voicetotext/ui/app.py`**

```python
"""Assemble QApplication + MainWindow + a real pipeline."""
from __future__ import annotations

from PySide6.QtWidgets import QApplication

from voicetotext.asr.parakeet import load_default as load_asr
from voicetotext.audio.sources import FileSource, MicSource
from voicetotext.pipeline.orchestrator import Pipeline
from voicetotext.translate.nllb import load_default as load_mt
from voicetotext.ui.main_window import MainWindow


def build_app(argv, *, file_path=None, use_mic=False, src="auto", tgt="rus_Cyrl"):
    app = QApplication(argv)
    window = MainWindow()

    source = MicSource() if use_mic else FileSource(file_path, realtime=True)
    engine = load_asr()
    translator = load_mt()

    pipeline = Pipeline(
        source=source,
        engine=engine,
        translator=translator,
        src_lang=("eng_Latn" if src == "auto" else src),
        tgt_lang=tgt,
        on_partial=window.partial_ready.emit,   # thread -> Qt signal -> GUI
        on_line=window.line_ready.emit,
    )
    window.set_pipeline(pipeline)
    window.tgt_combo.setCurrentText(tgt)
    return app, window
```

- [ ] **Step 7: Write `voicetotext/main.py`**

```python
"""CLI entry point."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="voicetotext")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="transcribe/translate an audio file")
    group.add_argument("--mic", action="store_true", help="use the microphone")
    parser.add_argument("--src", default="auto", help="source FLORES code or 'auto'")
    parser.add_argument("--tgt", default="rus_Cyrl", help="target FLORES code")
    args = parser.parse_args()

    from voicetotext.ui.app import build_app

    app, window = build_app(
        sys.argv[:1], file_path=args.file, use_mic=args.mic, src=args.src, tgt=args.tgt
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Run the UI test**

Run: `uv run pytest tests/test_ui.py -v`
Expected: PASS (3 passed). If `qtbot` is missing, ensure `pytest-qt` is in the dev extra (Task 1) and re-run `uv sync --extra dev`. On headless CI set `QT_QPA_PLATFORM=offscreen`.

- [ ] **Step 9: Commit**

```bash
git add voicetotext/ui/ voicetotext/main.py tests/conftest.py tests/test_ui.py
git commit -m "feat: transcript UI, main window, and application wiring"
```

---

### Task 10: End-to-end verification and benchmark script

Downloads the real models once, verifies the whole pipeline on a real audio file, and adds a benchmark script to pick adaptive defaults. This is the milestone acceptance gate.

**Files:**
- Create: `scripts/bench_rtf.py`
- Create: `scripts/download_models.py`
- Test: (manual/integration — run the real app)

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `scripts/download_models.py` — CLI that calls `ensure_model` for all three specs with console progress.
  - `scripts/bench_rtf.py` — feeds a WAV through `ParakeetEngine`, prints ASR real-time factor and mean NLLB ms/sentence.

- [ ] **Step 1: Write `scripts/download_models.py`**

```python
"""Download all models for first-run / CI setup."""
from voicetotext.models import registry
from voicetotext.models.download import ensure_model


def main() -> int:
    for spec in registry.ALL.values():
        path = ensure_model(spec, progress=print)
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Download the models**

Run: `uv run python scripts/download_models.py`
Expected: three models downloaded into the app data dir; each prints a "ready" line. This is ~1.3 GB and takes minutes. If a `repo_id` is wrong (see Task 4/6 implementer notes), fix the registry and re-run.

- [ ] **Step 3: Write `scripts/bench_rtf.py`**

```python
"""Measure ASR real-time-factor and MT latency on the host machine."""
import sys
import time
import wave

import numpy as np

from voicetotext.asr.parakeet import load_default as load_asr
from voicetotext.audio.resample import to_mono_16k
from voicetotext.translate.nllb import load_default as load_mt


def _load_wav(path):
    with wave.open(path, "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[width]
    data = np.frombuffer(raw, dtype=dtype)
    if ch > 1:
        data = data.reshape(-1, ch)
    return to_mono_16k(data, rate)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bench_rtf.py path/to/audio.wav")
        return 2
    samples = _load_wav(sys.argv[1])
    audio_seconds = len(samples) / 16000

    engine = load_asr()
    t0 = time.perf_counter()
    events = []
    for i in range(0, len(samples), 1600):
        events += engine.accept(samples[i : i + 1600])
    events += engine.flush()
    asr_wall = time.perf_counter() - t0
    print(f"ASR: {audio_seconds:.1f}s audio in {asr_wall:.2f}s  RTF={asr_wall/audio_seconds:.3f}")

    finals = [e for e in events if hasattr(e, "t_end")]
    tr = load_mt()
    sentences = [e.text for e in finals] or ["Hello, how are you?"]
    t0 = time.perf_counter()
    for s in sentences:
        tr.translate(s, "eng_Latn", "rus_Cyrl")
    mt_wall = time.perf_counter() - t0
    print(f"MT: {len(sentences)} sentences in {mt_wall:.2f}s  {1000*mt_wall/len(sentences):.0f} ms/sentence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the full test suite including integration**

Run: `uv run pytest -v`
Expected: all tests PASS now that models exist (the `integration`-marked NLLB and Parakeet tests execute and pass). If `test_speech_fixture_produces_a_final` fails only because `tests/fixtures/speech_16k.npy` is absent, record a real recording of a known phrase into that path (16 kHz mono float32 `np.save`) and re-run, or deselect it: `-k "not speech_fixture"`.

- [ ] **Step 5: Benchmark**

Run: `uv run python scripts/bench_rtf.py <some-speech>.wav`
Expected: prints RTF (should be well below 1.0 — target ≤0.3 on this M4 Pro) and ms/sentence for MT. Record the numbers; if RTF > 0.5, raise `partial_interval_s` in `ParakeetEngine` defaults.

- [ ] **Step 6: Run the real app on a file**

Run: `uv run voicetotext --file <some-speech>.wav --src eng_Latn --tgt rus_Cyrl`
Expected: a window opens; pressing **Start** plays the file through the pipeline and transcript lines with translations appear, with a grey partial row updating live. Verify a mic run too: `uv run voicetotext --mic --tgt rus_Cyrl` (grant mic permission on macOS).

- [ ] **Step 7: Commit**

```bash
git add scripts/
git commit -m "feat: model download and benchmark scripts; e2e verification"
```

---

## Self-Review

**Spec coverage** (design doc §§2–8, plus IMPLEMENTATION_SPEC):
- Parakeet ASR via sherpa-onnx (offline + VAD simulated streaming) → Task 6. ✓
- NLLB-200 translation via CTranslate2, per-sentence, FLORES codes → Tasks 3, 5. ✓
- Anti-flicker policy (finals translated, partials not) → Task 8 (asserted in tests). ✓
- 16 kHz mono float32 contract + resampling → Task 2, used everywhere. ✓
- Microphone capture (both OSes) → Task 7. ✓
- Transcript window with .txt/.srt export + live partial row + level meter → Task 9. ✓
- First-run model download into platformdirs data dir, disk check → Tasks 1, 4. ✓
- Threaded architecture (queues, Qt signals to GUI) → Tasks 8, 9. ✓
- Language pickers (source auto + target remembered) → Task 9 (persistence of "last target" wired via `config.save_settings` is a small follow-up; combo defaults to rus_Cyrl). NOTE below.
- **Deferred (own plans, stated in scope):** system-audio capture, subtitle overlay, Seamless pack, packaging (.exe/.dmg), macOS audio helper. ✓ (intentionally out of scope)

**Follow-up gap noted (not a blocker for this plan):** persisting the last-used target language via `config.save_settings`/`load_settings` is defined (Task 1) but only wired to a default in Task 9. Wiring `on_toggle` to save the chosen target and the window constructor to load it is a one-line addition; left for the UI-polish plan to avoid expanding this plan's UI scope.

**Placeholder scan:** No "TBD"/"add error handling"-style placeholders. Two explicit *implementer verification notes* (registry `repo_id`s in Task 4; sherpa-onnx VAD attribute names in Task 6) are deliberate — they flag the two facts that must be confirmed against the installed package/live HF at build time, with concrete verification commands. The `speech_16k.npy` fixture is optional and its absence degrades gracefully (documented in Tasks 9–10).

**Type consistency:** `Event`, `PartialTranscript`, `FinalTranscript` used identically across Tasks 6/8/9. `TranslatedLine(source, translation, t_start, t_end)` consistent Task 8→9. `Translator.translate(text, src, tgt)` consistent Tasks 5/8. `ASREngine.accept/flush` consistent Tasks 6/8. `AudioSource.start(on_audio)/stop` consistent Tasks 7/8/9. `to_mono_16k`, `TARGET_RATE` consistent Tasks 2/6/7/10.

**Scope:** Single cohesive deliverable — a runnable transcribe+translate app for mic and file input. Focused enough for one implementation pass.
