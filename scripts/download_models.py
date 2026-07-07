"""Download all models for first-run / CI setup."""
from voicetotext.models import registry
from voicetotext.models.download import ensure_model


def main() -> int:
    for spec in registry.ALL.values():
        path = ensure_model(spec, progress=print)
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
