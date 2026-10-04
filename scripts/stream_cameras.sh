#!/usr/bin/env bash
# Serve every processed dataset video as a looping RTSP camera via MediaMTX:
#   data/processed/<dataset>/<camera>.mp4  ->  rtsp://localhost:8554/<camera>
# Videos are pre-encoded by the adapters (1 s GOP, no B-frames), so ffmpeg only remuxes (-c copy).
#
# Usage: scripts/stream_cameras.sh [start|stop|status] [camera ...]
set -euo pipefail
cd "$(dirname "$0")/.."

cfg() { uv run --quiet python scripts/cfg.py "$1"; }
PROCESSED=$(cfg settings.paths.processed_dir)
RTSP_BASE=$(cfg settings.streaming.rtsp_base_url)
RUN_DIR="$(cfg settings.paths.cache_dir)/streams"
mkdir -p "$RUN_DIR"

cmd=${1:-start}; shift || true
wanted=("$@")

videos() {
  for f in "$PROCESSED"/*/*.mp4; do
    [ -e "$f" ] || continue
    cam=$(basename "$f" .mp4)
    if [ ${#wanted[@]} -gt 0 ] && [[ ! " ${wanted[*]} " =~ " $cam " ]]; then continue; fi
    echo "$cam $f"
  done
}

stop() {
  for pidf in "$RUN_DIR"/*.pid; do
    [ -e "$pidf" ] || continue
    pid=$(cat "$pidf"); cam=$(basename "$pidf" .pid)
    if kill "$pid" 2>/dev/null; then echo "stopped $cam"; fi
    rm -f "$pidf"
  done
}

status() {
  for pidf in "$RUN_DIR"/*.pid; do
    [ -e "$pidf" ] || { echo "no streams running"; return; }
    cam=$(basename "$pidf" .pid)
    if kill -0 "$(cat "$pidf")" 2>/dev/null; then echo "up   $RTSP_BASE/$cam"; else echo "dead $cam (see $RUN_DIR/$cam.log)"; fi
  done
}

start() {
  command -v ffmpeg >/dev/null || { echo "ffmpeg not found (sudo apt install ffmpeg)"; exit 1; }
  local n=0
  while read -r cam f; do
    pidf="$RUN_DIR/$cam.pid"
    if [ -e "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then echo "already up $cam"; continue; fi
    nohup ffmpeg -nostdin -hide_banner -loglevel warning -re -stream_loop -1 -i "$f" \
      -c copy -f rtsp -rtsp_transport tcp "$RTSP_BASE/$cam" >"$RUN_DIR/$cam.log" 2>&1 &
    echo $! >"$pidf"
    echo "streaming $RTSP_BASE/$cam  <-  $f"
    n=$((n + 1))
  done < <(videos)
  [ $n -gt 0 ] || [ -n "$(ls "$RUN_DIR"/*.pid 2>/dev/null)" ] || { echo "no processed videos in $PROCESSED (run the adapters)"; exit 1; }
  echo "open in VLC: Media > Open Network Stream > $RTSP_BASE/<camera>   (stop: $0 stop)"
}

case "$cmd" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 [start|stop|status] [camera ...]"; exit 2 ;;
esac
