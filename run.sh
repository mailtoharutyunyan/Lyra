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

echo "==> Syncing environment (first run downloads ~300 MB of packages)"
uv sync --extra dev

echo "==> Ensuring models are downloaded (first run: ~1.3 GB)"
uv run python scripts/download_models.py

# Default to system-audio capture translated to Russian if no args are given.
if [ "$#" -eq 0 ]; then
  set -- --system --tgt rus_Cyrl
fi

echo "==> Launching Lyra: $*"
exec uv run voicetotext "$@"
