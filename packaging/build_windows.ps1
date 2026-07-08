# Build the Windows onedir app and Inno Setup installer.
# Usage:  pwsh packaging\build_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not $env:LYRA_EDITION) { $env:LYRA_EDITION = "base" }
$AppName = if ($env:LYRA_EDITION -eq "extended") { "Lyra Extended" } else { "Lyra" }
$Version = if ($env:VERSION) { $env:VERSION } else { "0.1.0" }

Write-Host "==> Syncing deps for edition: $($env:LYRA_EDITION)"
if ($env:LYRA_EDITION -eq "extended") { uv sync --extra dev --extra seamless } else { uv sync --extra dev }
uv add --dev pyinstaller

Write-Host "==> PyInstaller build ($AppName)"
uv run pyinstaller packaging\app.spec --noconfirm --distpath dist --workpath build

# --- Code signing (optional): sign the exe if a cert is configured ---
if ($env:WINDOWS_CERT_THUMBPRINT) {
    Write-Host "==> Signing executable"
    & signtool sign /sha1 $env:WINDOWS_CERT_THUMBPRINT /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 "dist\$AppName\$AppName.exe"
} else {
    Write-Host "==> No WINDOWS_CERT_THUMBPRINT: unsigned build (SmartScreen will warn)"
}

Write-Host "==> Building installer with Inno Setup"
& iscc "/DVersion=$Version" "/DAppName=$AppName" packaging\installer.iss

Write-Host "==> Done: dist\installer\$AppName-$Version-setup.exe"
