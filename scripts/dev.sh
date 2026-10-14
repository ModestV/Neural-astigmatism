#!/usr/bin/env bash
# Запускает бэкенд (порт 8000) и фронтенд Vite (порт 5173) одной командой.
# Ctrl+C останавливает оба. Зависимости должны быть уже установлены (см. README).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/backend/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Не найден $PYTHON — создайте окружение: cd backend && python3 -m venv .venv" >&2
  exit 1
fi

trap 'kill 0' EXIT
(cd "$ROOT/backend" && "$PYTHON" -m uvicorn app.main:app --reload --port 8000) &
(cd "$ROOT/frontend" && npm run dev) &
wait
