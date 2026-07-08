"""Frozen-app entry point. When launched with no CLI args (double-click), open the
GUI defaulting to system audio; otherwise pass through to the normal CLI."""
import sys

from voicetotext.main import main

if __name__ == "__main__":
    # No forced source: the app opens with an in-window source picker.
    raise SystemExit(main())
