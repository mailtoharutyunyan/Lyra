"""CLI entry point."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="voicetotext")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="transcribe/translate an audio file")
    group.add_argument("--mic", action="store_true", help="use the microphone")
    parser.add_argument("--src", default="auto", help="source FLORES code or 'auto'")
    parser.add_argument("--tgt", default="rus_Cyrl", help="target FLORES code")
    args = parser.parse_args()

    from voicetotext.ui.app import build_app

    app, window = build_app(
        sys.argv[:1], file_path=args.file, use_mic=args.mic, src=args.src, tgt=args.tgt
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
