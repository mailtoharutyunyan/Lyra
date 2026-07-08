"""CLI entry point."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="voicetotext")
    # Source is optional — the app has an in-window picker. These just preselect it.
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--file", help="preselect an audio file")
    group.add_argument("--mic", action="store_true", help="preselect the microphone")
    group.add_argument("--system", action="store_true",
                       help="preselect system audio (whatever is playing on this computer)")
    parser.add_argument("--src", default="auto", help="source FLORES code or 'auto'")
    parser.add_argument("--tgt", default="rus_Cyrl", help="target FLORES code")
    args = parser.parse_args()

    from voicetotext.ui.app import build_app

    app, window = build_app(
        sys.argv[:1], file_path=args.file, use_mic=args.mic, use_system=args.system,
        src=args.src, tgt=args.tgt
    )
    if window is None:  # setup was cancelled (models not downloaded)
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
