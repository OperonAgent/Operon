<#
.SYNOPSIS
    Operon one-command installer for Windows.

.DESCRIPTION
    git clone https://github.com/OWNER/operon.git
    cd operon
    powershell -ExecutionPolicy Bypass -File install.ps1

    Flags are forwarded to install.py:
      -Full        every optional feature (voice, db, ...)
      -NoVenv      install into the current environment
      -NoBrowser   skip the Chromium browser binary

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Full
#>

param(
    [switch]$Full,
    [switch]$NoVenv,
    [switch]$NoBrowser,
    [switch]$Check
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# ── Find a suitable Python (>=3.9) ────────────────────────────────────────────
$py = $null
foreach ($cand in @("python", "python3", "py")) {
    $cmd = Get-Command $cand -ErrorAction SilentlyContinue
    if ($cmd) {
        $okver = & $cand -c "import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $py = $cand; break }
    }
}

if (-not $py) {
    Write-Host "X Python 3.9+ not found. Install it from https://python.org/downloads/" -ForegroundColor Red
    Write-Host "  Or from the Microsoft Store: search 'Python 3.12'"
    exit 1
}

Write-Host "-> Using $(& $py --version)" -ForegroundColor Cyan

# ── Forward flags to install.py ───────────────────────────────────────────────
$args = @()
if ($Full)      { $args += "--full" }
if ($NoVenv)    { $args += "--no-venv" }
if ($NoBrowser) { $args += "--no-browser" }
if ($Check)     { $args += "--check" }

& $py install.py @args
exit $LASTEXITCODE
