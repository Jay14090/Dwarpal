#!/usr/bin/env bash
# Native Linux alternative to webcam_publish.ps1: publish /dev/video0 to rtsp://localhost:8554/webcam.
# (Not usable inside WSL2: use scripts/webcam_publish.ps1 on Windows instead.)
set -euo pipefail
DEV=${1:-/dev/video0}
URL=${2:-rtsp://localhost:8554/webcam}
FPS=${FPS:-15}
exec ffmpeg -hide_banner -loglevel warning -f v4l2 -framerate "$FPS" -i "$DEV" \
  -c:v libx264 -preset ultrafast -tune zerolatency -g "$FPS" -bf 0 -pix_fmt yuv420p -an \
  -f rtsp -rtsp_transport tcp "$URL"
