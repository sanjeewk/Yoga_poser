#!/usr/bin/env bash
set -euo pipefail
trap 'kill 0' EXIT
cd "$(dirname "$0")/.."

if ! command -v npm >/dev/null 2>&1; then
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
fi

(backend/.venv/bin/uvicorn backend.app.main:app --reload --port 8000) &
(cd frontend && npm run dev) &

wait
