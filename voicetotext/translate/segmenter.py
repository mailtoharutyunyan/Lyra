"""Sentence splitting and language-code mapping (FLORES-200)."""
from __future__ import annotations

import re

# Parakeet TDT v3 2-letter output -> FLORES-200 codes used by NLLB-200.
LANG_CODES: dict[str, str] = {
    "bg": "bul_Cyrl", "hr": "hrv_Latn", "cs": "ces_Latn", "da": "dan_Latn",
    "nl": "nld_Latn", "en": "eng_Latn", "et": "est_Latn", "fi": "fin_Latn",
    "fr": "fra_Latn", "de": "deu_Latn", "el": "ell_Grek", "hu": "hun_Latn",
    "it": "ita_Latn", "lv": "lvs_Latn", "lt": "lit_Latn", "mt": "mlt_Latn",
    "pl": "pol_Latn", "pt": "por_Latn", "ro": "ron_Latn", "sk": "slk_Latn",
    "sl": "slv_Latn", "es": "spa_Latn", "sv": "swe_Latn", "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl", "hy": "hye_Armn",
}

# terminators: . ! ? … (keep the terminator with the sentence)
_SENT_RE = re.compile(r".*?[.!?…]+|.+$", re.DOTALL)


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [m.group(0).strip() for m in _SENT_RE.finditer(text)]
    return [p for p in parts if p]


def to_flores(code: str) -> str:
    if "_" in code:  # already a FLORES code
        return code
    return LANG_CODES[code]
