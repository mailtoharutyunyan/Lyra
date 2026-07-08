"""Assemble QApplication + first-run setup + MainWindow with an on-demand pipeline."""
from __future__ import annotations

from PySide6.QtWidgets import QApplication

from voicetotext.audio.sources import FileSource, MicSource
from voicetotext.audio.system import make_system_source
from voicetotext.pipeline.orchestrator import Pipeline
from voicetotext.ui.main_window import MainWindow


def build_app(argv, *, file_path=None, use_mic=False, use_system=False,
              src="auto", tgt="rus_Cyrl"):
    app = QApplication(argv)

    # First run: download models with a progress screen before showing the app.
    from voicetotext.ui.setup import ensure_models_ready
    if not ensure_models_ready():
        return app, None

    window = MainWindow()

    # Load the heavy models once and reuse across start/stop and source changes.
    cache: dict = {}

    def _engines():
        if "asr" not in cache:
            from voicetotext.asr.parakeet import load_default as load_asr
            from voicetotext.translate.nllb import load_default as load_mt
            cache["asr"] = load_asr()
            cache["mt"] = load_mt()
        return cache["asr"], cache["mt"]

    def make_pipeline(kind, path, src_lang, tgt_lang):
        if kind == "system":
            source = make_system_source()
        elif kind == "file":
            source = FileSource(path, realtime=True)
        else:
            source = MicSource()
        engine, translator = _engines()
        return Pipeline(
            source=source, engine=engine, translator=translator,
            src_lang=("eng_Latn" if src_lang == "auto" else src_lang),
            tgt_lang=tgt_lang,
            on_partial=window.partial_ready.emit,
            on_line=window.line_ready.emit,
            on_level=window.level_ready.emit,
        )

    window.set_pipeline_factory(make_pipeline)

    # Apply CLI-provided defaults to the pickers.
    if file_path:
        window._file_path = file_path
        window.source_combo.blockSignals(True)  # don't pop the file dialog
        window.source_combo.setCurrentIndex(2)
        import os
        window.source_combo.setItemText(2, f"📂  {os.path.basename(file_path)}")
        window.source_combo.blockSignals(False)
    elif use_system:
        window.source_combo.setCurrentIndex(1)
    elif use_mic:
        window.source_combo.setCurrentIndex(0)
    _select(window.tgt_combo, tgt)
    if src != "auto":
        _select(window.src_combo, src)
    return app, window


def _select(combo, code) -> None:
    for i in range(combo.count()):
        if combo.itemData(i) == code:
            combo.setCurrentIndex(i)
            return
