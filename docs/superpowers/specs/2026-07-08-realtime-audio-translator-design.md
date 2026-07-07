# Real-Time Local Audio Translator — Design

**Date:** 2026-07-08
**Status:** Approved by user (pending spec review)
**Working name:** VoiceToText

## 1. Summary

A Windows + macOS desktop app that captures **system audio** (any video/call playing on the machine) or the **microphone**, transcribes speech **fully locally**, translates it **fully locally** into a user-chosen target language, and displays the result two ways:

- a **floating subtitle overlay** (always-on-top, draggable, click-through toggle), and
- a **transcript history window** (scrolling original + translation, export to .txt/.srt).

Source and target languages are selectable per session. Distributed as a Windows installer (.exe) and a macOS disk image (.dmg). For **personal / free use** (non-commercial model licenses are acceptable).

Non-goals (v1): speech output (dubbing), mobile, cloud engines, per-app audio capture, Linux, auto-update.

## 2. Model / engine choices

| Role | Engine | Why | Size | License |
|---|---|---|---|---|
| ASR (default) | **NVIDIA Parakeet TDT 0.6B v3** (int8 ONNX) via **sherpa-onnx** | 25 languages (En, Ru, Uk + European set), more accurate than Whisper on those languages, 10–20× real-time on CPU, no hallucinations on silence/music | ~487 MB | CC-BY-4.0 |
| ASR (extended, optional) | **SeamlessM4T v2 large** (PyTorch) | ~100 source languages incl. **Armenian speech input**; runs in "delayed captions" mode on VAD segments | ~5 GB, needs 16 GB+ RAM | CC-BY-NC-4.0 |
| VAD | **Silero VAD** via sherpa-onnx | Segments speech, gates ASR, provides endpointing | <3 MB | MIT |
| Translation | **NLLB-200 distilled 600M** (int8) via **CTranslate2** | 200 languages incl. Armenian (`hye_Armn`), Russian (`rus_Cyrl`); ~50–300 ms/sentence on CPU | ~640 MB | CC-BY-NC-4.0 |

Whisper is deliberately excluded (user experience: poor quality, known hallucination and low-resource-language weaknesses).

Language capability matrix shown in UI:
- Parakeet engine: 25 source languages → any of ~200 NLLB target languages.
- Seamless pack installed: ~100 source languages (incl. Armenian) → target via Seamless S2TT directly (Armenian→X) with NLLB fallback for uncovered pairs.

## 3. Architecture

Single Python process (PySide6/Qt UI) + a tiny native Swift helper binary on macOS for system audio. Pipeline stages run on worker threads connected by queues; UI updates via Qt signals.

```
AudioSource ──► RingBuffer(16 kHz mono f32) ──► VAD ──► ASREngine ──► SentenceSplitter ──► Translator ──► UI
```

### 3.1 Audio capture (`app/audio/`)
- **Microphone (both OSes):** `sounddevice` InputStream; capture at device native rate, downmix to mono, resample to 16 kHz (`soxr` or `resample_poly`).
- **Windows system audio:** WASAPI loopback via `PyAudioWPatch` — zero setup, no drivers, no permission prompt.
- **macOS system audio:** bundled Swift helper using **Core Audio process taps** (macOS 14.4+, audio-only TCC permission), streaming PCM to Python over a pipe; ScreenCaptureKit path as fallback for macOS 13–14.3. BlackHole documented only as a last-resort manual fallback — never required.
- Common interface: `AudioSource.start(callback)` delivering 16 kHz mono float32 frames + RMS level for the UI meter.

### 3.2 ASR (`app/asr/`)
- Interface: `ASREngine.feed(frames) -> list[Event]` where Event is `Partial(text)` or `Final(text, t0, t1)`.
- **ParakeetEngine** (default): sherpa-onnx `OfflineRecognizer.from_transducer` + `VoiceActivityDetector` (Silero). "Simulated streaming" pattern (maintainer-endorsed): while VAD reports active speech, re-decode the accumulated segment every ~300 ms → `Partial`; on VAD endpoint (min_silence ≈ 0.15–0.25 s, max_speech ≈ 8 s) decode once more → `Final`. Segment length capped so re-decode cost stays bounded.
- **SeamlessEngine** (optional pack): offline decode of VAD-finalized segments only (no partials); emits `Final` events a few seconds behind live. Direct speech→translated-text where the pair is supported.
- Language: user-pinned source language (recommended default UI state) or auto-detect.

### 3.3 Translation (`app/mt/`)
- Interface: `Translator.translate(text, src, tgt) -> str`.
- **NLLBTranslator**: CTranslate2 int8, FLORES-200 codes, `inter_threads=1, intra_threads=4`.
- Policy (anti-flicker, research-backed): translate **only finalized sentences**. Partials are displayed untranslated in grey/italic. Finalized ASR text is sentence-split (Parakeet outputs punctuation; simple punctuation splitter, `wtpsplit` if needed) and each sentence translated once, then immutable.
- If translation backlog exceeds ~2 s of text, show originals rather than lag the captions; surface a status indicator.

### 3.4 UI (`app/ui/`)
- **Overlay window:** frameless, translucent, always-on-top, draggable; click-through toggle; settings for font size, opacity, colors, 1–3 lines shown; modes: translation-only / original+translation stacked. Partial text styled grey/italic, solidifies on finalization.
- **Main window:** transcript table (time, original, translation) with copy and **export to .txt / .srt**; language pickers (source: engine-filtered list + auto-detect, target: NLLB list, remembers last); audio source picker with **live level meter**; engine/model manager panel; start/stop.
- **Tray/menu-bar icon:** start/stop, show/hide overlay, quit.

### 3.5 Model manager (`app/models/`)
- First-run guided download via `huggingface_hub.snapshot_download` (parallel, resumable) into `platformdirs.user_data_dir("VoiceToText")/models`; Qt progress dialog; free-disk-space check (size + 20% headroom); checksum verification (built into HF hub).
- Seamless pack offered only if RAM ≥ 16 GB; shows size warning (~5 GB).

### 3.6 Settings (`app/config.py`)
- JSON in `platformdirs.user_config_dir`; stores last languages, overlay geometry/appearance, audio device, engine choice.

## 4. Error handling

- Every stage pushes status/errors to a UI status bar (device disconnected, model load failure, helper binary missing/permission denied, translation backlog).
- Audio-capture failures are **loud**: explicit dialog + log file (silent no-text failure is the top complaint in competing apps).
- macOS permission flow: onboarding screen that requests/checks the system-audio permission and mic permission with clear instructions.
- Watchdog: if ASR thread stalls, pipeline restarts the engine and drops stale audio ("catch-up" policy — skip old audio rather than fall behind live).

## 5. Testing

- Unit tests per stage with recorded WAV fixtures: feed known audio → assert transcript contains expected phrases; translator tests with fixed sentences; sentence-splitter and anti-flicker policy tests are pure-logic tests.
- Integration: dev-mode `--play-file x.wav` audio source that pipes a file through the real pipeline end-to-end without live capture.
- Manual test matrix: Win10/11 laptop (CPU-only), macOS (M-series) — system audio + mic, at least En→Ru, Ru→En, En→Hy pairs.

## 6. Packaging & distribution

- **PyInstaller onedir** builds per OS with `collect_all` for `sherpa_onnx`, `ctranslate2`, `onnxruntime`. Never onefile (slow start, AV false positives, 2 GB limit). Torch is NOT in the base bundle — the Seamless pack downloads a torch runtime separately if/when chosen.
- **Windows:** Inno Setup single setup.exe, per-user install (no UAC). Unsigned for v1 (users click through SmartScreen); Azure Artifact Signing later if wanted.
- **macOS:** arm64-only .app; `dmgbuild` for the .dmg. Info.plist keys: `NSMicrophoneUsageDescription`, `NSAudioCaptureUsageDescription`. Hardened-runtime codesign + notarization required for public distribution (Apple Developer, $99/yr) — dev/local builds run with ad-hoc signing on the developer's machine.
- Expected installer sizes: ~150–300 MB (models excluded, downloaded at first run).
- Updates: manual "check for updates" pointing at GitHub Releases (v1).

## 7. Repository layout

```
voicetotext/
  audio/        # sources: mic, wasapi_loopback, mac_tap, blackhole, file (dev); resample.py
  asr/          # engine.py (VAD + OfflineRecognizer loop), models.py (download/verify)
  translate/    # base.py (Translator protocol), nllb.py, seamless.py (optional)
  pipeline/     # orchestrator.py (threads, queues, message dataclasses), segmenter.py
  ui/           # overlay, main window, tray, onboarding, model download dialogs
  helpers/mac/  # Swift Core Audio tap helper (own build step, shipped in .app)
packaging/      # pyinstaller specs, InnoSetup .iss, dmgbuild settings
scripts/        # bench_rtf.py (first-run hardware benchmark)
tests/          # unit + integration, audio fixtures per language
docs/
```

Code-level API details (verified sherpa-onnx/CTranslate2 usage, tuned VAD parameters,
milestone acceptance criteria) live in the companion document `docs/IMPLEMENTATION_SPEC.md`;
this design doc governs scope and decisions, the implementation spec governs the how.

## 8. Build order (high level)

1. Pipeline core on files: file audio source → Parakeet → NLLB → console output (proves the ML stack).
2. Mic capture + minimal transcript window.
3. Overlay window + anti-flicker display policy.
4. Windows WASAPI loopback source.
5. macOS Swift audio helper.
6. Model download manager + onboarding.
7. Seamless extended pack.
8. Packaging (.exe, .dmg).

## 9. Risks

- **sherpa-onnx M-series real-time factor unverified** — benchmark in step 1; fallback: FluidInference CoreML Parakeet or smaller model.
- **PyObjC/ScreenCaptureKit flakiness** — mitigated by using a compiled Swift helper instead of PyObjC.
- **macOS system-audio permission requires signed binary** — dev machine OK with local signing; public distribution needs Apple Developer account (user decision deferred).
- **Seamless on CPU (Windows) is not real-time** — scoped as "delayed captions" and hardware-gated; set expectations in UI.
