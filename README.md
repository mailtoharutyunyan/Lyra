# Lyra

Live, private subtitles and translation for anything your computer hears — fully offline.

Lyra turns any audio your Mac or PC plays — videos, calls, or your mic — into live
subtitles translated into your language, fully on-device. Powered by NVIDIA Parakeet
speech recognition and Meta's NLLB translation: offline, private, no accounts and no
virtual audio cables. An optional edition adds speech recognition for 100+ languages,
including Armenian.

## Features

- **Hears anything** — microphone, an audio file, or **system audio** (whatever is
  playing on the computer), with no virtual audio cable to install.
- **Fully local** — speech recognition and translation run on-device. No cloud, works
  offline, nothing leaves your machine.
- **Real-time** — recognized speech is translated sentence-by-sentence and collected
  in a running history; ~6× faster than real-time on Apple Silicon.
- **Pick your languages** — choose the source (or **Auto-detect**) and target from
  human-readable menus; changes apply live, no restart.
- **Two speech models** — *Standard* (Parakeet, 25 languages, fast) and *Extended*
  (SeamlessM4T, 100+ languages incl. Armenian).
- **Export** the transcript to `.srt` or `.txt`.

## Editions

Both are fully self-contained — an end user needs **no Python, uv, or pip** installed.

| Edition | Size | Languages |
|---|---|---|
| **Lyra** | small | 25 languages (incl. English, Russian, Ukrainian, European) + system audio |
| **Lyra Extended** | large | adds Armenian and 100+ languages (bundles PyTorch/SeamlessM4T) |

Install the app, open it, and it downloads its models on first run through a setup
screen — no terminal, no configuration.

## Run from source (developers)

Requires [uv](https://docs.astral.sh/uv/); Python is managed for you.

```bash
./run.sh                 # opens the app; pick source, model, and languages in-window
```

Models download on first run into the project `models/` folder (git-ignored).
`run.sh` includes the optional `seamless` extra so the Extended model works too.

## How it works

```
audio (mic / file / system) → 16 kHz mono → Silero VAD (finds speech) →
  speech recognition (Parakeet or SeamlessM4T) → NLLB translation → history
```

- **Silero VAD** and **NLLB** always run (voice detection + translation); the *Speech
  model* menu chooses the recognizer.
- Only finalized sentences are translated, so the history stays stable (no flicker).
- With **Auto** source: Standard detects the language of the recognized text; Extended
  uses SeamlessM4T's speech-to-text translation directly.

## Building installers

CI builds both editions for macOS (`.dmg`) and Windows (`.exe`) on version tags
(`.github/workflows/build.yml`). Code signing and notarization activate automatically
when the corresponding repository secrets are set.

## Licensing

Parakeet is CC-BY-4.0 (commercial use OK). NLLB-200 and SeamlessM4T are CC-BY-NC-4.0
(non-commercial). Lyra is intended for personal / non-commercial use; the translation
backend sits behind an interface so it can be swapped for a permissively-licensed model.

## Development

```bash
uv run pytest -m "not integration"   # fast unit tests (no models needed)
uv run pytest                        # full suite (requires downloaded models)
```
