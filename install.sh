#!/usr/bin/env bash
#
# install.sh — Operon one-command installer for macOS / Linux.
#
#   git clone https://github.com/OperonAgent/Operon.git
#   cd operon
#   ./install.sh
#
# Flags are forwarded to install.py:
#   ./install.sh --full        # every optional feature (voice, db, …)
#   ./install.sh --no-venv     # install into the current environment
#   ./install.sh --no-browser  # skip the Chromium browser binary
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Find a suitable Python (>=3.9) ────────────────────────────────────────────
PY=""
for cand in python3.12 python3.11 python3.10 python3.9 python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)' 2>/dev/null; then
      PY="$cand"
      break
    fi
  fi
done

if [ -z "$PY" ]; then
  echo "✗ Python 3.9+ not found. Install it from https://python.org/downloads/"
  echo "  macOS:  brew install python@3.12"
  echo "  Linux:  sudo apt install python3 python3-venv python3-pip"
  exit 1
fi

echo "→ Using $($PY --version) at $(command -v "$PY")"
exec "$PY" install.py "$@"
