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
