#!/usr/bin/env bash
# Launch Lyra (the desktop app) for testing.
#
# Usage:
#   ./run.sh                          # system audio -> Russian (default)
#   ./run.sh --mic --tgt eng_Latn     # microphone -> English
#   ./run.sh --file talk.wav --tgt rus_Cyrl
#   ./run.sh --system --tgt hye_Armn  # system audio -> Armenian
#
# Target languages use FLORES-200 codes: eng_Latn, rus_Cyrl, ukr_Cyrl, hye_Armn, ...
set -euo pipefail
cd "$(dirname "$0")"

# Include the "seamless" extra so PyTorch is present and the Extended model works.
# (uv re-syncs to the requested extras on every run; without this it removes torch.)
echo "==> Syncing environment (first run downloads packages incl. PyTorch)"
uv sync --extra dev --extra seamless

# Models download from inside the app on first run (a setup screen with progress),
# so no separate download step is needed here.

echo "==> Launching Lyra"
# No source flag: the app opens with an in-window picker (Microphone / System audio /
# Open file) and language menus. Pass flags to preselect, e.g. --system --tgt hye_Armn.
exec uv run voicetotext "$@"
