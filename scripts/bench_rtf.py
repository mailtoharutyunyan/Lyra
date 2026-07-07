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
