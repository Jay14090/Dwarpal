#!/usr/bin/env bash
# `make demo`: everything for the demo in one terminal (Ctrl-C stops it all).
#   infra (Postgres+pgvector, MediaMTX) -> migrations -> cache check -> API + engine -> dashboard
# The webcam is published from Windows separately: scripts/webcam_publish.ps1 (see docs/demo_script.md).
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=${PORT:-8000}
WEB_PORT=${WEB_PORT:-3000}
RUN="uv run"
# the webcam stream is absent until it is published: keep OpenCV/FFmpeg reconnect noise out of the log
export OPENCV_FFMPEG_LOGLEVEL=${OPENCV_FFMPEG_LOGLEVEL:--8} OPENCV_LOG_LEVEL=${OPENCV_LOG_LEVEL:-ERROR}

say() { printf '\033[36m[demo]\033[0m %s\n' "$*"; }

[ -f .env ] || cp .env.example .env
say "starting Postgres + MediaMTX"
docker compose up -d >/dev/null
$RUN python scripts/wait_for_db.py
$RUN alembic upgrade head >/dev/null

say "checking cached cameras"
$RUN python - <<'PY'
from app.core.config import get_config
from app.pipeline.cache import cache_path
c = get_config()
for cam in c.cameras.cameras:
    if not cam.enabled:
        continue
    if cam.run_mode == "cached" and not cache_path(c.settings.paths.cache_dir, cam.id).is_file():
        print(f"  ! {cam.id}: no detection cache -> it will run realtime (slow). Run `make index CAMERAS={cam.id}`.")
    else:
        print(f"  ok {cam.id} ({cam.run_mode})")
PY

if [ ! -d frontend/node_modules ]; then say "installing dashboard dependencies"; (cd frontend && npm install --no-audit --no-fund); fi
if [ ! -f frontend/.next/BUILD_ID ] || [ -n "$(find frontend/src -newer frontend/.next/BUILD_ID -type f | head -1)" ]; then
  say "building dashboard"; (cd frontend && NEXT_TELEMETRY_DISABLED=1 npm run build >/dev/null)
fi

pids=()
cleanup() { say "stopping"; kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

say "starting API + video engine on http://localhost:$PORT"
$RUN uvicorn app.main:app --host 0.0.0.0 --port "$PORT" &
pids+=($!)
say "starting dashboard on http://localhost:$WEB_PORT"
(cd frontend && NEXT_TELEMETRY_DISABLED=1 npx next start -p "$WEB_PORT") &
pids+=($!)

until curl -sf "localhost:$PORT/health" >/dev/null; do sleep 1; done
say "ready: open http://localhost:$WEB_PORT/live   (API docs: http://localhost:$PORT/docs)"
say "webcam: run scripts/webcam_publish.ps1 in Windows PowerShell; demo steps: docs/demo_script.md"
wait -n "${pids[@]}"
