#!/bin/bash
set -e
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/3.12/bin:$PATH"
trap 'status=$?; if [ "$status" -ne 0 ]; then printf "\nSetup stopped (exit %s). Check the message above.\n" "$status"; read -r -p "Press Return to close."; fi' EXIT
cd "$(dirname "$0")"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  printf 'This experimental app currently requires an Apple Silicon Mac.\n'; read -r -p 'Press Return to close.'; exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  PYTHON="$(command -v python3.12 || true)"
  if [[ -z "$PYTHON" ]]; then
    printf 'Python 3.12 is required. Install it from https://www.python.org/downloads/macos/ or with: brew install python@3.12\nThen open this file again.\n'
    read -r -p 'Press Return to close.'; exit 1
  fi
  printf 'Creating Python environment...\n'
  "$PYTHON" -m venv .venv
fi
if [[ ! -f .venv/.hunyuan-deps-ready ]]; then
  printf 'Installing Python dependencies. This can take several minutes.\n'
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/.hunyuan-deps-ready
fi
printf '\nStarting Hunyuan3D at http://127.0.0.1:7864/\nKeep this window open.\n'
.venv/bin/python local_app.py
