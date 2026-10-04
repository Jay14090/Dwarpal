"""Render a short video with ground-truth boxes drawn on top (adapter sanity check).

Example: python scripts/render_gt_overlay.py meva meva_g328 --start 92 --seconds 10
Writes data/clips/gt_<camera>_<start>s.mp4 by default.
"""

from __future__ import annotations

import argparse
import colorsys
import sys
from pathlib import Path

import cv2

from app.core.config import get_config
from app.datasets.common import GroundTruth


def color_for(gid: int) -> tuple[int, int, int]:
    h = (gid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def render(video: Path, gt: GroundTruth, out: Path, start_s: float, seconds: float) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or gt.fps
    first = round(start_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    boxes = gt.by_frame()
    drawn = 0
    for i in range(round(seconds * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        idx = first + i
        for gid, (x1, y1, x2, y2), cls_ in boxes.get(idx, []):
            c = color_for(gid)
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, c, 2)
            cv2.putText(frame, f"{cls_[0].upper()}{gid}", (p1[0], max(12, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)  # fmt: skip
            drawn += 1
        cv2.putText(frame, f"{gt.camera} f={idx}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)  # fmt: skip
        writer.write(frame)
    writer.release()
    cap.release()
    return drawn


def first_annotated_second(gt: GroundTruth) -> float:
    frames = sorted({int(r[0]) for r in gt.rows})
    return frames[0] / gt.fps if frames else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("camera")
    parser.add_argument("--start", type=float, help="seconds; default: first annotated frame")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    paths = get_config().settings.paths
    ddir = paths.processed_dir / args.dataset
    gt = GroundTruth.load(ddir / "gt" / f"{args.camera}.json")
    start = args.start if args.start is not None else first_annotated_second(gt)
    out = args.out or paths.clips_dir / f"gt_{args.camera}_{int(start)}s.mp4"
    n = render(ddir / f"{args.camera}.mp4", gt, out, start, args.seconds)
    print(f"wrote {out} ({n} boxes drawn, start={start:.1f}s)")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
