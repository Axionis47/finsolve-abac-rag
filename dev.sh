#!/bin/sh
# Development helper for this machine.
#
# Why this exists: this repo lives on an iCloud-synced Desktop, and macOS keeps
# re-applying the UF_HIDDEN flag to files under .venv. Python's site setup
# silently skips hidden .pth files, which breaks the editable install and with
# it the `finrag` console script. Clearing the flag after every sync fixes it.
#
# Usage:
#   ./dev.sh sync    install or update the venv, then repair the flags
#   ./dev.sh test    run the full test suite
#   ./dev.sh eval    run the naive baseline eval
set -e
UV="$HOME/.local/bin/uv"
REPAIR() { chflags nohidden .venv/lib/python3.12/site-packages/*.pth 2>/dev/null || true; }

case "$1" in
  sync) "$UV" sync --group dev; REPAIR ;;
  test) REPAIR; "$UV" run --no-sync pytest -q ;;
  eval) REPAIR; "$UV" run --no-sync finrag eval --mode "${2:-naive}" --rung "${3:-r0}" ;;
  *) echo "usage: ./dev.sh sync|test|eval [mode] [rung]"; exit 1 ;;
esac
