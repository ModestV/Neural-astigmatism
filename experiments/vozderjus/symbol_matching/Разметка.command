#!/bin/zsh
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
exec "$REPO_ROOT/backend/.venv/bin/python" "$SCRIPT_DIR/annotate.py"
