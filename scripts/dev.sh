#!/usr/bin/env bash
set -euo pipefail
trap 'kill 0' EXIT

(cd backend && . .venv/bin/activate && uvicorn app.main:app --reload --port 8000) &
BACK_PID=$!
(cd frontend && npm run dev) &
FRONT_PID=$!

wait
