"""Lightweight source-language detection for the 'Auto' option (Standard model).

Detects the language of recognized text and maps it to a FLORES-200 code that
NLLB understands. Falls back to English when detection is uncertain.
"""
from __future__ import annotations

# langdetect ISO-639-1 code -> FLORES-200 code (only the ones we translate from).
_ISO_TO_FLORES = {
    "en": "eng_Latn", "ru": "rus_Cyrl", "uk": "ukr_Cyrl", "de": "deu_Latn",
    "fr": "fra_Latn", "es": "spa_Latn", "it": "ita_Latn", "pt": "por_Latn",
    "pl": "pol_Latn", "nl": "nld_Latn", "el": "ell_Grek", "cs": "ces_Latn",
    "ro": "ron_Latn", "sv": "swe_Latn", "bg": "bul_Cyrl", "hr": "hrv_Latn",
    "hu": "hun_Latn", "fi": "fin_Latn", "da": "dan_Latn", "hy": "hye_Armn",
    "ka": "kat_Geor", "ar": "arb_Arab", "fa": "pes_Arab", "tr": "tur_Latn",
}


def detect_flores(text: str, default: str = "eng_Latn") -> str:
    text = text.strip()
    if len(text) < 2:
        return default
    try:
        from langdetect import detect

        return _ISO_TO_FLORES.get(detect(text), default)
    except Exception:
        return default
