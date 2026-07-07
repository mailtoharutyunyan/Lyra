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


def test_level_callback_receives_rms_for_each_chunk():
    chunks = [np.full(1600, 0.5, dtype=np.float32), np.zeros(1600, dtype=np.float32)]
    source = FakeAudioSource(chunks)
    engine = ScriptedASREngine(script={})
    levels = []
    pipe = Pipeline(source=source, engine=engine, translator=RecordingTranslator(),
                    src_lang="eng_Latn", tgt_lang="rus_Cyrl",
                    on_partial=lambda _: None, on_line=lambda _: None,
                    on_level=levels.append)
    pipe.run_file_blocking()
    assert len(levels) == 2
    assert levels[0] > 0.4   # RMS of constant 0.5
    assert levels[1] == 0.0  # silence
