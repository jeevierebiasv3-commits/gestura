# build_gestura.ps1 — one-shot build of the Gestura desktop app.
#
#   powershell -ExecutionPolicy Bypass -File build_gestura.ps1
#
# Produces dist\Gestura\Gestura.exe. Run from the project root.

$ErrorActionPreference = "Stop"
$py = ".\venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "ERROR: venv not found at $py — run this from the project root." -ForegroundColor Red
    exit 1
}

Write-Host "==> Ensuring PyInstaller is installed in the venv…" -ForegroundColor Cyan
& $py -m pip install --quiet "pyinstaller>=6.3" ; if (-not $?) { exit 1 }

Write-Host "==> Cleaning previous build…" -ForegroundColor Cyan
if (Test-Path build) { Remove-Item build -Recurse -Force }
if (Test-Path dist)  { Remove-Item dist  -Recurse -Force }

Write-Host "==> Building (this takes several minutes — TensorFlow is large)…" -ForegroundColor Cyan
& $py -m PyInstaller gestura.spec --noconfirm ; if (-not $?) { exit 1 }

$exe = "dist\Gestura\Gestura.exe"
if (Test-Path $exe) {
    $size = "{0:N0} MB" -f ((Get-ChildItem dist\Gestura -Recurse | Measure-Object Length -Sum).Sum / 1MB)
    Write-Host ""
    Write-Host "==> SUCCESS. Built $exe  (total $size)" -ForegroundColor Green
    Write-Host "    Test it:   .\$exe"
    Write-Host "    Ship it:   zip the whole  dist\Gestura\  folder."
} else {
    Write-Host "==> Build finished but $exe not found — check the PyInstaller output above." -ForegroundColor Red
    exit 1
}
