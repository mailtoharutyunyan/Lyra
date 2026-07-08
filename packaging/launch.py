"""Frozen-app entry point. When launched with no CLI args (double-click), open the
GUI defaulting to system audio; otherwise pass through to the normal CLI."""
import sys

from voicetotext.main import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Double-clicked bundle: default to system-audio capture.
        sys.argv += ["--system"]
    raise SystemExit(main())
