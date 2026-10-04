"""Track clips cut from a camera's video file, with privacy blur burned in (P7 + P10).

Frames are decoded with OpenCV, the head region of every person who must stay anonymous is blurred
using the camera's cached detections, and the result is piped to ffmpeg (H.264, faststart). Without
cached detections a clip cannot be anonymised and is refused (unless an admin unblurs).
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2

from app.pipeline.cache import TrackCache
from app.privacy import blur_heads

# frame index -> person boxes to blur in that frame
BlurFn = Callable[[int], list[tuple[float, float, float, float]]]


class ClipError(RuntimeError):
    pass


def blur_fn(cache: TrackCache, keep_track_id: int | None) -> BlurFn:
    """Blur every person in the frame except `keep_track_id` (the identified person the clip is about)."""

    def boxes(frame: int) -> list[tuple[float, float, float, float]]:
        return [
            t.xyxy for t in cache.tracks_at(frame) if t.is_person and t.track_id != keep_track_id
        ]

    return boxes


def cut_clip(
    src: Path,
    out: Path,
    start_frame: int,
    end_frame: int,
    blur: BlurFn | None,
    pad_s: float = 1.0,
    max_s: float = 122.0,
    head_fraction: float = 0.24,
) -> Path:
    if shutil.which("ffmpeg") is None:
        raise ClipError("ffmpeg not found")
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ClipError(f"cannot open {src}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or end_frame + 1
        first = max(0, int(start_frame - pad_s * fps))
        last = min(n - 1, int(end_frame + pad_s * fps), first + int(max_s * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part.mp4")
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", f"{fps:.3f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]  # fmt: skip
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        try:
            for f in range(first, last + 1):
                ok, img = cap.read()
                if not ok:
                    break
                if blur is not None:
                    boxes = blur(f)
                    if boxes:
                        img = blur_heads(img, boxes, head_fraction)
                proc.stdin.write(img.tobytes())
            proc.stdin.close()
            err = proc.stderr.read().decode() if proc.stderr else ""
            if proc.wait(timeout=120) != 0:
                raise ClipError(f"ffmpeg failed: {err[-300:]}")
        finally:
            if proc.poll() is None:
                proc.kill()
        tmp.replace(out)
        return out
    finally:
        cap.release()
