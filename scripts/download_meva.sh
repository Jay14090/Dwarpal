#!/usr/bin/env bash
# Download the configured MEVA subset (one date-hour slot, 2-3 outdoor cameras) plus annotations.
# Public bucket, no AWS account needed. Lists sizes first; total is ~0.5 GB for the default slot.
# Config: settings.datasets.meva (config/settings.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."

cfg() { uv run --quiet python scripts/cfg.py "$1"; }
command -v aws >/dev/null || { echo "aws CLI not found: 'uv tool install awscli' or 'sudo apt install awscli'"; exit 1; }

BUCKET=$(cfg settings.datasets.meva.bucket)
VIDEO_PREFIX=$(cfg settings.datasets.meva.video_prefix)
ANN_PREFIX=$(cfg settings.datasets.meva.annotation_prefix)
CLIP=$(cfg settings.datasets.meva.clip_prefix)
CAMERAS=$(cfg settings.datasets.meva.cameras)
CONFIRM_GB=$(cfg settings.datasets.download_confirm_gb)
OUT="$(cfg settings.paths.raw_dir)/meva"
mkdir -p "$OUT/videos" "$OUT/annotations"

echo "Listing $BUCKET/$VIDEO_PREFIX for clip $CLIP, cameras: $CAMERAS"
listing=$(aws s3 ls --no-sign-request "$BUCKET/$VIDEO_PREFIX/")
total=0
files=()
for cam in $CAMERAS; do
  line=$(grep -E " ${CLIP}[^ ]*\.${cam}\.r13\.avi$" <<<"$listing" | head -1 || true)
  [ -n "$line" ] || { echo "  no video for $cam in $CLIP"; exit 1; }
  size=$(awk '{print $3}' <<<"$line"); name=$(awk '{print $4}' <<<"$line")
  awk -v n="$name" -v s="$size" 'BEGIN{printf "  %-55s %8.1f MB\n", n, s/1e6}'
  total=$((total + size)); files+=("$name")
done
awk -v t="$total" 'BEGIN{printf "Total: %.2f GB\n", t/1e9}'
if awk -v t="$total" -v c="$CONFIRM_GB" 'BEGIN{exit !(t/1e9 > c)}' && [ "${1:-}" != "--yes" ]; then
  echo "Over ${CONFIRM_GB} GB: re-run with --yes to confirm."; exit 1
fi

for name in "${files[@]}"; do
  [ -s "$OUT/videos/$name" ] && { echo "have $name"; continue; }
  aws s3 cp --no-sign-request --only-show-errors "$BUCKET/$VIDEO_PREFIX/$name" "$OUT/videos/"
done
for cam in $CAMERAS; do
  aws s3 cp --no-sign-request --only-show-errors --recursive "$BUCKET/$ANN_PREFIX/" "$OUT/annotations/" \
    --exclude "*" --include "${CLIP}*.${cam}.*"
done
# Site map used to pick gate/parking views.
[ -s "$OUT/phase2-known-facility-site-map.pdf" ] || \
  aws s3 cp --no-sign-request --only-show-errors "$BUCKET/phase2-known-facility-site-map.pdf" "$OUT/"
echo "done -> $OUT"
