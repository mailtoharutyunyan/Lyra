# Build the Windows onedir app and Inno Setup installer.
# Usage:  pwsh packaging\build_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$AppName = "Lyra"
$Version = if ($env:VERSION) { $env:VERSION } else { "0.1.0" }

Write-Host "==> PyInstaller build"
uv run pyinstaller packaging\app.spec --noconfirm --distpath dist --workpath build

# --- Code signing (optional): sign the exe if a cert is configured ---
if ($env:WINDOWS_CERT_THUMBPRINT) {
    Write-Host "==> Signing executable"
    & signtool sign /sha1 $env:WINDOWS_CERT_THUMBPRINT /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 "dist\$AppName\$AppName.exe"
} else {
    Write-Host "==> No WINDOWS_CERT_THUMBPRINT: unsigned build (SmartScreen will warn)"
}

Write-Host "==> Building installer with Inno Setup"
& iscc "/DVersion=$Version" packaging\installer.iss

Write-Host "==> Done: dist\installer\$AppName-$Version-setup.exe"
