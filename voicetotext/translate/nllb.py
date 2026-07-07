"""NLLB-200 translation via CTranslate2 (int8, CPU)."""
from __future__ import annotations

from pathlib import Path

from voicetotext.models.download import ensure_model
from voicetotext.models.registry import NLLB


class NLLBTranslator:
    def __init__(self, model_dir: str | Path, intra_threads: int = 4) -> None:
        import ctranslate2
        import transformers

        self._model_dir = str(model_dir)
        self._ct = ctranslate2.Translator(
            self._model_dir, device="cpu", inter_threads=1, intra_threads=intra_threads
        )
        self._tok = transformers.AutoTokenizer.from_pretrained(self._model_dir)

    def translate(self, text: str, src: str, tgt: str) -> str:
        text = text.strip()
        if not text:
            return ""
        self._tok.src_lang = src
        tokens = self._tok.convert_ids_to_tokens(self._tok.encode(text))
        results = self._ct.translate_batch([tokens], target_prefix=[[tgt]])
        out = results[0].hypotheses[0]
        if out and out[0] == tgt:  # strip target-language prefix token
            out = out[1:]
        return self._tok.decode(self._tok.convert_tokens_to_ids(out))


def load_default() -> "NLLBTranslator":
    model_dir = ensure_model(NLLB)
    return NLLBTranslator(model_dir)
