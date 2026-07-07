# VoiceToText — Implementation Specification

Local, offline, real-time speech-to-text + translation desktop app.
Python, cross-platform (Windows + macOS), CPU-first (Apple Silicon and average x86 laptops).

This document is the single source of truth for implementation. All technical claims below
were verified against upstream docs/repos in July 2026; items marked **[UNCERTAIN]** need
benchmarking or verification on target hardware before being relied on.

---

## 1. Architecture overview

```
 Audio source (mic / system audio)
        │  native rate, 1-2 ch, float32
        ▼
 Capture thread ──► ring buffer ──► downmix to mono + resample to 16 kHz (soxr)
        │
        ▼
 Silero VAD (sherpa-onnx VoiceActivityDetector, 512-sample / 32 ms hops)
        │
        ├─ while segment open: every 300–500 ms re-decode accumulated segment
        │       └► PARTIAL transcript (unstable, shown grey/italic, never translated*)
        │
        └─ on segment close (silence ≥ min_silence_duration or max_speech_duration hit):
                └► FINAL transcript (sherpa-onnx OfflineRecognizer, Parakeet TDT 0.6B v3 int8)
                        │
                        ▼
                 Sentence splitter (punctuation-based; Parakeet v3 outputs punct+caps)
                        │
                        ▼
                 NLLB-200 600M int8 via CTranslate2 (per finalized sentence)
                        │
                        ▼
                 UI: append immutable transcript line + translation line
```

Key architectural decision (verified): **Parakeet TDT 0.6B v3 has NO true streaming variant
in sherpa-onnx** — it is an offline (non-streaming) transducer. The maintainer-endorsed
pattern for pseudo-real-time is **VAD segmentation + repeated re-decoding of the open
segment ("simulated streaming")**. Do NOT use `OnlineRecognizer` for this model.
References:
- Model docs: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html
- Open feature requests confirming no true streaming: https://github.com/k2-fsa/sherpa-onnx/issues/2918 , https://github.com/k2-fsa/sherpa-onnx/issues/3573
- Official example to mirror: `python-api-examples/vad-with-non-streaming-asr.py` and
  `simulate-streaming-paraformer-microphone.py` in https://github.com/k2-fsa/sherpa-onnx

There are no streaming (online) sherpa-onnx models with multilingual European coverage
(streaming zipformers exist only for en/zh/ko/bn/zh-en), so this architecture is not
optional — it is the only viable one for the 25-language requirement.

---

## 2. Models: exact artifacts, sizes, licenses

| Model | Artifact | Download | Unpacked / runtime | License |
|---|---|---|---|---|
| ASR default | `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2` from https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models | **487 MB** | ~640 MB (encoder 622 + decoder 12 + joiner 6.1 MB) | CC-BY-4.0 (commercial OK) |
| VAD | `silero_vad.onnx` from same release page | **0.6 MB** (v5: 2.3 MB) | trivial | MIT |
| Translation | `JustFrederik/nllb-200-distilled-600M-ct2-int8` (HF) — model.bin **622.6 MB** + ~22 MB tokenizer files. Alternatively convert `facebook/nllb-200-distilled-600M` yourself with `ct2-transformers-converter --quantization int8`. | ~645 MB | ~700 MB RAM | **CC-BY-NC-4.0 (non-commercial)** |
| Extended ASR+translation (optional) | `facebook/seamless-m4t-v2-large` (HF), 2.3B params | ~9 GB fp32 / ~4.6 GB fp16 | ~5.8 GB RAM fp16, ~11.6 GB fp32 | **CC-BY-NC-4.0 (non-commercial)** |

Notes:
- Only int8 is published for Parakeet v3 in sherpa-onnx releases; that is what we want anyway.
- Parakeet v3 languages (25): bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk.
  It auto-detects language; it outputs punctuation and capitalization (needed for sentence splitting).
- **Licensing asymmetry**: Parakeet path is commercial-friendly; NLLB and Seamless are
  non-commercial. Keep NLLB/Seamless as separate optional downloads and surface the license
  to the user. If the app ever goes commercial, translation must be swapped (e.g., OPUS-MT
  models are permissive) — isolate translation behind an interface (`Translator` protocol).
- SeamlessStreaming (true simultaneous speech→text) is an additional 8 GB checkpoint and is
  **not real-time on CPU** (Meta recommends GPU). Do not build on it. See §7.

Model storage: download on first run into a per-user data dir
(`platformdirs.user_data_dir("VoiceToText")`), with SHA/size verification and resumable
download. Never bundle models in the installer.

---

## 3. ASR pipeline (sherpa-onnx) — implementation details

Python package: `sherpa-onnx` (PyPI). Verified APIs:

```python
import sherpa_onnx

recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=".../encoder.int8.onnx",
    decoder=".../decoder.int8.onnx",
    joiner=".../joiner.int8.onnx",
    tokens=".../tokens.txt",
    num_threads=4,               # tune; leave cores for MT + UI
    model_type="nemo_transducer",
)

vad_config = sherpa_onnx.VadModelConfig()
vad_config.silero_vad.model = ".../silero_vad.onnx"
vad_config.silero_vad.threshold = 0.5
vad_config.silero_vad.min_silence_duration = 0.15   # s; lower = snappier finalization
vad_config.silero_vad.min_speech_duration = 0.25    # s
vad_config.silero_vad.max_speech_duration = 8.0     # s; forces cut on long monologue
vad_config.sample_rate = 16000
vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)
```

Processing loop (runs in a dedicated ASR thread, consuming the 16 kHz mono queue):

1. Feed audio to VAD in 512-sample (32 ms) windows: `vad.accept_waveform(chunk)`.
2. **Partials**: keep your own `current_segment` buffer of samples since VAD reported
   speech start. Every `PARTIAL_INTERVAL = 0.4 s` (and only if the segment is open),
   decode it:
   ```python
   stream = recognizer.create_stream()
   stream.accept_waveform(16000, current_segment)
   recognizer.decode_stream(stream)
   partial_text = stream.result.text
   ```
   Emit `PartialTranscript(text)` to the UI. Cost is bounded because segments are capped
   at `max_speech_duration`; Parakeet int8 decodes ~10–20x faster than real time on a
   modern CPU, so re-decoding ≤8 s of audio every 0.4 s is affordable.
3. **Finals**: `while not vad.empty(): segment = vad.front; vad.pop()` — decode the
   closed segment once, emit `FinalTranscript(text, t_start, t_end)`, clear partial state.
4. Language: expose a "language hint" setting but default to auto (model handles it).

Threading model: capture thread (audio callback → `queue.Queue`), ASR thread, MT thread,
UI main thread. Communicate via queues with dataclass messages. onnxruntime and
CTranslate2 both release the GIL during inference, so plain threads are fine — no
multiprocessing needed.

Performance targets (verify on hardware):
- i7-12700K reference: RTF ≈ 0.05 int8 (source: https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai)
- **[UNCERTAIN]** M-series via onnxruntime CPU: expect RTF 0.05–0.15 (no published numbers;
  a CoreML port reaches 110x RT on M4 Pro, which bounds what the silicon can do).
- **[UNCERTAIN]** Average 4–8 core x86 laptop: expect RTF 0.1–0.3. If partial re-decoding
  falls behind on weak machines, degrade gracefully: increase `PARTIAL_INTERVAL`, then
  disable partials (finals-only mode). Make this adaptive (measure decode wall time).

---

## 4. Translation (NLLB-200 via CTranslate2)

```python
import ctranslate2, transformers

translator = ctranslate2.Translator(model_dir, device="cpu",
                                    inter_threads=1, intra_threads=4)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, src_lang="eng_Latn")

def translate(text: str, src: str, tgt: str) -> str:
    tokenizer.src_lang = src
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    result = translator.translate_batch([tokens], target_prefix=[[tgt]])
    out = result[0].hypotheses[0][1:]          # strip target-lang token
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(out))
```

- Language codes are FLORES-200. **Verified supported**: `hye_Armn` (Armenian),
  `rus_Cyrl` (Russian), `eng_Latn`, `ukr_Cyrl`, `deu_Latn`, `fra_Latn`, `spa_Latn`, etc.
  Full list on the model card: https://huggingface.co/entai2965/nllb-200-distilled-600M-ctranslate2
  Map Parakeet's 2-letter language output → FLORES codes with a static dict.
- 512-token input limit → translate **per sentence**, never whole paragraphs.
- Sentence splitting: Parakeet emits punctuation, so split finals on `[.!?…]` + heuristics
  (abbreviations, decimals). If quality is insufficient, use `wtpsplit` — but start simple.
- Expected latency: **[UNCERTAIN]** roughly 50–300 ms per short sentence on a modern laptop
  CPU with int8 (no authoritative published per-sentence CPU numbers; benchmark).
  Batch: if multiple sentences finalize together, pass them as one `translate_batch` call.
- Armenian quality caveat: NLLB handles `hye_Armn` but it is a lower-resource direction;
  set expectations in docs/UI.

### Caption stability policy (anti-flicker)

Grounded in the simultaneous-translation literature
(re-translation: https://arxiv.org/abs/1912.03393 ; dynamic masking: https://arxiv.org/pdf/2006.00249 ;
MeetDot system: https://arxiv.org/pdf/2109.09577):

- **Default mode ("stable")**: translate only FINAL sentences. Translations are immutable
  once rendered. Partial ASR text is displayed untranslated in a visually distinct style.
- **Optional mode ("live", v2 feature)**: retranslate the open partial, but
  (a) debounce — retranslate only if ≥ 4 new source words AND ≥ 700 ms since last call;
  (b) mask the last 2–4 tokens of the MT output (don't render them) until confirmed by
      the next retranslation — this is the mask-k / dynamic-masking trick;
  (c) never revise translation lines that are more than one segment old.
- Never let the MT queue back up: if a new final arrives while a partial translation is
  in flight, drop the stale partial job.

---

## 5. Audio capture

Target format everywhere: **16 kHz, mono, float32 in [-1, 1]**.
Capture at the device's native rate (typically 44.1/48 kHz, 1–2 ch), then:
downmix `mono = frames.mean(axis=1)` → resample with **`soxr`** (`soxr.resample`, best
quality/speed) — do this before VAD (Silero strictly requires 16 kHz, 512-sample hops).
(sherpa-onnx `accept_waveform` can resample internally, but VAD cannot — so resample once,
up front.)

### Microphone (both OSes)
`sounddevice` (PortAudio). `sd.InputStream(samplerate=native, channels=..., dtype="float32",
callback=...)`; callback pushes copies into the queue (never do work in the callback).
Device enumeration + a device picker in settings; handle default-device changes.

### Windows system audio (loopback)
Primary: **PyAudioWPatch** (https://github.com/s0d3s/PyAudioWPatch) — PortAudio fork with
WASAPI loopback; wheels for Python 3.7–3.13; loopback devices appear as extra *input*
devices (use `p.get_default_wasapi_loopback()` helper / iterate `get_loopback_device_info_generator()`).
Stream format is the mixer format (usually 48 kHz stereo int16/float32) → downmix+resample.
Fallback/alternative: `soundcard` lib with `include_loopback=True` (use git master; the
released version has a Windows bug). Per-process capture (later, nice-to-have): ProcTap
(https://github.com/m96-chan/ProcTap).

### macOS system audio
No pure-Python first-party path. Implement behind a `SystemAudioSource` interface with
this priority order:
1. **Core Audio process tap** (macOS 14.4+): `CATapDescription` +
   `AudioHardwareCreateProcessTap`. Reference implementation: AudioCap
   (https://github.com/insidegui/AudioCap). Build a ~200-line Swift helper binary that
   writes raw PCM (or WAV stream) to stdout; Python spawns it via `subprocess` and reads
   the pipe. Requires TCC audio-recording permission (helper triggers the prompt; app must
   be signed for the permission to stick).
2. **ScreenCaptureKit** (macOS 13+): also possible from the same Swift helper. Do NOT try
   pure PyObjC — SCStream via PyObjC is known-broken on macOS 15
   (https://github.com/ronaldoussoren/pyobjc/issues/647).
   Community wrappers exist (`macloop` on PyPI — Rust engine, CoreAudio+SCK backends;
   ProcTap SCK backend) — **[UNCERTAIN]** young projects; evaluate `macloop` first since it
   would remove the need for our own helper, but keep the Swift-helper design as the
   dependable path.
3. **BlackHole** virtual device fallback (works with plain `sounddevice`); document the
   Multi-Output Device setup for users on macOS < 14.4.

Mic on macOS: plain `sounddevice` (needs mic TCC permission).

---

## 6. UI / app shell

(Choose per existing project conventions; if greenfield:)
- **PySide6** recommended: mature, native-feeling on both OSes, good threading story
  (signals/slots for queue→UI marshaling), packagable with PyInstaller/briefcase.
- Main window: scrolling caption view. Each utterance = one block: source line (final,
  black) + translation line (accent color); one "live" block at the bottom for the open
  partial (grey italic). Auto-scroll unless user scrolled up.
- Controls: source selector (mic / system audio / device), ASR language hint (auto default),
  target translation language, translation on/off, font size, always-on-top compact
  "caption bar" mode.
- Settings persisted with `QSettings` or a JSON in the user config dir.
- First-run model download UI with progress + license acknowledgment for NLLB/Seamless.

---

## 7. SeamlessM4T v2 (optional extended-language module)

Purpose: languages outside Parakeet's 25 (primary motivation: **Armenian**).
Verified: `facebook/seamless-m4t-v2-large` supports Armenian (`hye`) as speech+text
*source* and text *target* (no Armenian speech output — irrelevant for us).

- Mode of use: **offline segment translation only** — feed VAD-finalized segments to
  `SeamlessM4Tv2Model` (transformers) task `s2tt` (speech → translated text) or ASR.
  A 3–8 s segment takes a few seconds on CPU: "delayed captions", not partials. This is
  acceptable for the extended-language tier; make latency expectations clear in the UI.
- **Do not use SeamlessStreaming**: 8 GB checkpoint, not real-time on CPU (GPU recommended
  upstream), non-commercial license, heavy `seamless_communication`/fairseq2 dependency.
- Gating: require ≥ 16 GB RAM to enable this module (fp16/bf16 load ≈ 5.8 GB RAM
  — use fp32 only if numerical issues appear on CPU, at ~11.6 GB). On Apple Silicon try
  `torch` MPS; **[UNCERTAIN]** MPS speedup for this model is unbenchmarked — implement CPU
  first, MPS as a flag.
- Keep it in an optional pip extra (`pip install voicetotext[seamless]`) so the base app
  doesn't pull torch/transformers.

---

## 8. Project structure & milestones

```
voicetotext/
  audio/        sources.py (Mic, WasapiLoopback, MacTap, BlackHole), resample.py
  asr/          engine.py (VAD + OfflineRecognizer loop), models.py (download/verify)
  translate/    base.py (Translator protocol), nllb.py, seamless.py (optional)
  pipeline/     orchestrator.py (threads, queues, message dataclasses), segmenter.py
  ui/           main_window.py, caption_view.py, settings.py
  helpers/mac/  AudioTapHelper (Swift, built in CI, shipped in app bundle)
```

Milestones (each independently testable):
1. **M1 — Mic → live transcript**: sounddevice + soxr + VAD + Parakeet, partials+finals,
   minimal window. Acceptance: speak, see grey partial within ~0.5 s, black final at pause.
2. **M2 — Translation**: NLLB int8, per-sentence, stable mode. Acceptance: en→ru and
   ru→en captions; translation appears < 1 s after final.
3. **M3 — System audio**: PyAudioWPatch loopback (Win), Swift tap helper or macloop (mac),
   BlackHole fallback. Acceptance: transcribe a YouTube video with speakers muted... (via
   loopback, i.e., without mic).
4. **M4 — Robustness/perf**: adaptive partial interval, device hot-swap, long-run soak
   (2 h, stable memory), model download UX.
5. **M5 — Seamless module (optional)**: Armenian speech → English/Russian text.

Testing notes: keep a `tests/fixtures/` of short WAVs per language; unit-test the pipeline
by injecting file audio into the same queue the capture thread uses (no audio hardware in CI).
Benchmark script (`scripts/bench_rtf.py`) that reports ASR RTF and MT ms/sentence on the
host — run it on first launch to pick adaptive defaults.

Pinned dependencies to start from: `sherpa-onnx`, `sounddevice`, `soxr`, `numpy`,
`ctranslate2`, `transformers` (tokenizer only for NLLB), `platformdirs`, `PySide6`;
Windows-only: `PyAudioWPatch`; optional extra: `torch`, `sentencepiece` (Seamless).
