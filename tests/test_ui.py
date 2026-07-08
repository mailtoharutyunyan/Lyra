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
    assert w.windowTitle() == "Lyra"


def test_language_pickers_use_human_names_and_carry_codes(qtbot):
    from voicetotext.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    # target combo shows a readable name but carries the FLORES code as data
    labels = [w.tgt_combo.itemText(i) for i in range(w.tgt_combo.count())]
    codes = [w.tgt_combo.itemData(i) for i in range(w.tgt_combo.count())]
    assert "rus_Cyrl" not in labels          # no raw codes shown
    assert any("Russian" in x for x in labels)
    assert "rus_Cyrl" in codes                # code still available programmatically
    # source picker offers mic + system + file
    src_kinds = [w.source_combo.itemData(i) for i in range(w.source_combo.count())]
    assert set(src_kinds) == {"mic", "system", "file"}


def test_level_meter_import(qtbot):
    from voicetotext.ui.widgets import LevelMeter
    m = LevelMeter()
    qtbot.addWidget(m)
    m.set_level(0.3)  # must not raise


def test_language_change_applies_live_without_restart(qtbot):
    from voicetotext.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)

    class StubPipeline:
        def __init__(self): self.calls = []
        def set_languages(self, src, tgt): self.calls.append((src, tgt))

    stub = StubPipeline()
    w.set_pipeline(stub)
    w._running = True                      # simulate a running session
    # switch target language via the dropdown
    idx = next(i for i in range(w.tgt_combo.count())
               if w.tgt_combo.itemData(i) == "eng_Latn")
    w.tgt_combo.setCurrentIndex(idx)
    assert stub.calls and stub.calls[-1][1] == "eng_Latn"


def test_model_chooser_lists_both_engines(qtbot):
    from voicetotext.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    keys = [w.model_combo.itemData(i) for i in range(w.model_combo.count())]
    assert keys == ["parakeet", "seamless"]
