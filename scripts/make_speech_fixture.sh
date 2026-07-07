#!/usr/bin/env bash
# Generate the integration-test speech fixture from macOS text-to-speech.
# Produces tests/fixtures/speech_en_16k.wav (16 kHz mono, LEI16).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tests/fixtures
say -o /tmp/vtt_speech.aiff "Hello, how are you today? The weather is very nice."
afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/vtt_speech.aiff tests/fixtures/speech_en_16k.wav
echo "wrote tests/fixtures/speech_en_16k.wav"
