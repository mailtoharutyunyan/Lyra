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
