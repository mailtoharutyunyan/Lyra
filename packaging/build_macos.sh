#!/usr/bin/env bash
# Build the macOS .app and .dmg. Signing + notarization run only if the relevant
# secrets are present; otherwise it produces an unsigned build (fine for local use,
# but system-audio capture and Gatekeeper need signing for distribution).
set -euo pipefail
cd "$(dirname "$0")/.."

export LYRA_EDITION="${LYRA_EDITION:-base}"
if [ "$LYRA_EDITION" = "extended" ]; then APP_NAME="Lyra Extended"; else APP_NAME="Lyra"; fi
VERSION="${VERSION:-0.1.0}"

echo "==> Syncing deps for edition: $LYRA_EDITION"
if [ "$LYRA_EDITION" = "extended" ]; then
  uv sync --extra dev --extra seamless
else
  uv sync --extra dev
fi
uv add --dev pyinstaller

echo "==> PyInstaller build ($APP_NAME)"
uv run pyinstaller packaging/app.spec --noconfirm --distpath dist --workpath build

# --- Code signing (optional) ---
if [[ -n "${MACOS_SIGN_IDENTITY:-}" ]]; then
  echo "==> Codesigning with hardened runtime"
  codesign --force --deep --options runtime --timestamp \
    --entitlements packaging/entitlements.plist \
    --sign "$MACOS_SIGN_IDENTITY" "dist/${APP_NAME}.app"
else
  echo "==> No MACOS_SIGN_IDENTITY: producing an UNSIGNED .app (local use only)"
fi

echo "==> Building DMG"
uv run --with dmgbuild dmgbuild -s packaging/dmg_settings.py "$APP_NAME" "dist/${APP_NAME}-${VERSION}.dmg"

# --- Notarization (optional) ---
if [[ -n "${AC_API_KEY_ID:-}" && -n "${AC_API_ISSUER:-}" && -n "${AC_API_KEY_PATH:-}" ]]; then
  echo "==> Notarizing DMG"
  xcrun notarytool submit "dist/${APP_NAME}-${VERSION}.dmg" \
    --key "$AC_API_KEY_PATH" --key-id "$AC_API_KEY_ID" --issuer "$AC_API_ISSUER" --wait
  xcrun stapler staple "dist/${APP_NAME}-${VERSION}.dmg"
else
  echo "==> No notarization credentials: DMG is not notarized"
fi

echo "==> Done: dist/${APP_NAME}-${VERSION}.dmg"
