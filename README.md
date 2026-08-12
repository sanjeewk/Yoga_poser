# Yoga Poser

Real-time yoga pose detection. FastAPI + MediaPipe + scikit-learn backend, React + Vite frontend.

See `docs/superpowers/specs/2026-08-12-yoga-pose-detector-design.md` for design.

## Quick start

1. Python 3.10+ and Node 18+ required.
2. Backend: `python3 -m venv backend/.venv && source backend/.venv/bin/activate && pip install -r backend/requirements.txt`
3. Frontend: `cd frontend && npm install`
4. Train: see `backend/training/` README.
5. Run: `./scripts/dev.sh` → open http://localhost:5173
