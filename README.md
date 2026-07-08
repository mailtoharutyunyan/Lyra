# Lyra

Live, private subtitles and translation for anything your computer hears — fully offline.

Lyra turns any audio your Mac or PC plays — videos, calls, or your mic — into live
subtitles translated into your language, fully on-device. Powered by NVIDIA Parakeet
speech recognition and Meta's NLLB translation: offline, private, no accounts and no
virtual audio cables. An optional pack adds speech recognition for 100+ languages,
including Armenian.

## Features

- **Hears anything** — microphone, an audio file, or **system audio** (whatever is
  playing on the computer), with no virtual audio cable to install.
- **Fully local** — speech recognition (Parakeet TDT 0.6B v3 via sherpa-onnx) and
  translation (NLLB-200 via CTranslate2) run on-device. No cloud, works offline.
- **Real-time** — ~6× faster than real-time on Apple Silicon; live partial captions
  that settle into finalized, translated lines (no flicker).
- **Cross-platform** — macOS (system audio via macloop) and Windows (WASAPI loopback).
- **Extended languages (optional)** — a SeamlessM4T v2 pack adds Armenian and ~100
  other source languages for speech input.

## Requirements

- Python 3.12 (managed via [uv](https://docs.astral.sh/uv/))
- macOS 14.4+ (Apple Silicon) or Windows 10/11 (x64)

## Quick start

```bash
uv sync --extra dev                 # create the environment
uv run python scripts/download_models.py   # first-run model download (~1.3 GB)

uv run voicetotext --system --tgt rus_Cyrl # translate system audio -> Russian
uv run voicetotext --mic    --tgt eng_Latn # translate the microphone -> English
uv run voicetotext --file talk.wav --tgt rus_Cyrl
```

Target languages use FLORES-200 codes (`eng_Latn`, `rus_Cyrl`, `ukr_Cyrl`, `hye_Armn`, …).

### Optional: extended-language pack (Armenian etc.)

```bash
uv sync --extra seamless            # installs torch + tokenizer deps
```

Requires ≥ 16 GB RAM; runs in a slower "delayed captions" mode.

## How it works

```
audio (mic / file / system) → resample 16 kHz mono → Silero VAD →
  Parakeet ASR (simulated streaming) → sentence split → NLLB translation → captions
```

Only finalized sentences are translated (partials show untranslated), which keeps
captions stable instead of flickering.

## Building installers

CI builds a macOS `.dmg` and a Windows `.exe` on version tags
(`.github/workflows/build.yml`). Locally:

```bash
bash packaging/build_macos.sh       # -> dist/Lyra-<version>.dmg
pwsh packaging/build_windows.ps1    # -> dist/installer/Lyra-<version>-setup.exe
```

Code signing and notarization activate automatically when the corresponding secrets
are configured; unsigned builds work for local use.

## Licensing

Parakeet is CC-BY-4.0 (commercial use OK). NLLB-200 and SeamlessM4T are CC-BY-NC-4.0
(non-commercial). Lyra is intended for personal / non-commercial use; the translation
backend is behind an interface so it can be swapped for a permissively-licensed model.

## Development

```bash
uv run pytest -m "not integration"  # fast unit tests (no models needed)
uv run pytest                       # full suite (requires downloaded models)
```
