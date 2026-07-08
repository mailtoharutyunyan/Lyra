"""Human-readable language names for the UI, keyed by FLORES-200 code."""
from __future__ import annotations

# "Native · English" where a native name helps; English-only otherwise.
FLORES_NAME: dict[str, str] = {
    "eng_Latn": "English",
    "rus_Cyrl": "Русский · Russian",
    "ukr_Cyrl": "Українська · Ukrainian",
    "hye_Armn": "Հայերեն · Armenian",
    "deu_Latn": "Deutsch · German",
    "fra_Latn": "Français · French",
    "spa_Latn": "Español · Spanish",
    "ita_Latn": "Italiano · Italian",
    "por_Latn": "Português · Portuguese",
    "pol_Latn": "Polski · Polish",
    "nld_Latn": "Nederlands · Dutch",
    "swe_Latn": "Svenska · Swedish",
    "dan_Latn": "Dansk · Danish",
    "fin_Latn": "Suomi · Finnish",
    "ell_Grek": "Ελληνικά · Greek",
    "ces_Latn": "Čeština · Czech",
    "slk_Latn": "Slovenčina · Slovak",
    "slv_Latn": "Slovenščina · Slovenian",
    "hrv_Latn": "Hrvatski · Croatian",
    "bul_Cyrl": "Български · Bulgarian",
    "ron_Latn": "Română · Romanian",
    "hun_Latn": "Magyar · Hungarian",
    "est_Latn": "Eesti · Estonian",
    "lvs_Latn": "Latviešu · Latvian",
    "lit_Latn": "Lietuvių · Lithuanian",
    "mlt_Latn": "Malti · Maltese",
    "kat_Geor": "ქართული · Georgian",
    "azj_Latn": "Azərbaycan · Azerbaijani",
    "fas_Arab": "فارسی · Persian",
}

# Targets Lyra offers (must be NLLB-supported); order = display order.
TARGET_CODES = [
    "eng_Latn", "rus_Cyrl", "ukr_Cyrl", "hye_Armn",
    "deu_Latn", "fra_Latn", "spa_Latn", "ita_Latn", "por_Latn",
]

# Source options for the standard (Parakeet) model: auto-detect + its languages.
SOURCE_CODES = [
    "auto", "eng_Latn", "rus_Cyrl", "ukr_Cyrl", "deu_Latn",
    "fra_Latn", "spa_Latn", "ita_Latn", "por_Latn", "pol_Latn",
    "nld_Latn", "ell_Grek", "ces_Latn", "ron_Latn", "swe_Latn",
]

# Source options for the extended (SeamlessM4T) model — includes Armenian etc.
# No "auto": Seamless needs an explicit source language.
SEAMLESS_SOURCE_CODES = [
    "hye_Armn", "eng_Latn", "rus_Cyrl", "ukr_Cyrl", "kat_Geor",
    "deu_Latn", "fra_Latn", "spa_Latn", "ita_Latn", "por_Latn",
]

# Model options: (label, key). The key selects the ASR engine in app.py.
MODEL_OPTIONS = [
    ("Standard · 25 languages, fast", "parakeet"),
    ("Extended · 100+ languages incl. Armenian", "seamless"),
]


def name_for(code: str) -> str:
    if code == "auto":
        return "Detect automatically"
    return FLORES_NAME.get(code, code)
