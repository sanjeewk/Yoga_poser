#!/usr/bin/env bash
set -euo pipefail
trap 'kill 0' EXIT

(backend/.venv/bin/uvicorn backend.app.main:app --reload --port 8000) &
(cd frontend && npm run dev) &

wait
