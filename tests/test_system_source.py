import sys
import time

import numpy as np
import pytest

from voicetotext.audio.system import capabilities, make_system_source, SystemAudioUnavailable


def test_capabilities_reports_current_os():
    caps = capabilities()
    assert caps["os"] == sys.platform
    assert caps["method"] in {"macloop", "wasapi-loopback", "none"}
    assert isinstance(caps["system_audio"], bool)
    assert caps["notes"]


def test_make_source_returns_audiosource_or_raises():
    try:
        src = make_system_source()
        assert hasattr(src, "start") and hasattr(src, "stop")
        assert getattr(src, "streaming", False) is True
    except SystemAudioUnavailable as e:
        assert str(e)  # must carry guidance


def test_streaming_source_end_of_stream_does_not_drop_audio():
    """Regression: a live source whose start() returns immediately must still
    have all its audio processed; the end sentinel must wait for stop()."""
    from voicetotext.asr.events import FinalTranscript, PartialTranscript
    from voicetotext.pipeline.orchestrator import Pipeline
    from tests.fakes import StreamingFakeSource, ScriptedASREngine, RecordingTranslator

    chunks = [np.zeros(1600, dtype=np.float32) for _ in range(3)]
    source = StreamingFakeSource(chunks)
    engine = ScriptedASREngine(
        script={2: [FinalTranscript(text="Hello there.", t_start=0.0, t_end=1.0)]}
    )
    tr = RecordingTranslator()
    lines = []
    pipe = Pipeline(source=source, engine=engine, translator=tr,
                    src_lang="eng_Latn", tgt_lang="rus_Cyrl",
                    on_partial=lambda _: None, on_line=lines.append)
    pipe.start()
    time.sleep(0.3)  # let the streaming thread deliver all chunks
    pipe.stop()      # only now should the end sentinel fire
    assert [c[0] for c in tr.calls] == ["Hello there."]
    assert len(lines) == 1 and lines[0].translation == "[rus_Cyrl]Hello there."
