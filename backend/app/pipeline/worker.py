"""Per-camera worker: source -> (detect -> track | cache lookup) -> hooks -> annotate -> publish.

Realtime and cached cameras share this code path; only where the tracks come from differs,
so the dashboard looks identical in both modes (CLAUDE.md section 4).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2

from app.core.config import Camera
from app.pipeline.annotate import Annotator
from app.pipeline.cache import TrackCache
from app.pipeline.detector import Detector
from app.pipeline.frames import Frame, FrameResult, Track
from app.pipeline.sources import FrameSource
from app.pipeline.tracker import Tracker
from app.privacy import PrivacyPolicy

log = logging.getLogger(__name__)

# (camera_id, jpeg bytes, metadata) -> None. Must not block (the engine drops when full).
PublishFn = Callable[[str, bytes, dict[str, Any]], None]


class FrameHook(Protocol):
    """Later phases plug in here (Re-ID, global IDs, identity, attributes, rules)."""

    def __call__(self, result: FrameResult) -> None: ...


@dataclass
class RateMeter:
    window_s: float = 5.0
    _times: deque[float] = field(default_factory=deque)

    def tick(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._times.append(now)
        while self._times and now - self._times[0] > self.window_s:
            self._times.popleft()

    @property
    def rate(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


class CameraWorker:
    def __init__(
        self,
        camera: Camera,
        source: FrameSource,
        publish: PublishFn,
        annotator: Annotator,
        *,
        detector: Detector | None = None,
        tracker: Tracker | None = None,
        cache: TrackCache | None = None,
        hooks: list[FrameHook] | None = None,
        publish_fps: float = 15.0,
        jpeg_quality: int = 80,
        max_infer_fps: float = 15.0,
        privacy: PrivacyPolicy | None = None,
    ) -> None:
        if cache is None and (detector is None or tracker is None):
            raise ValueError(f"{camera.id}: needs a cache or a detector + tracker")
        self.camera = camera
        self.source = source
        self.publish = publish
        self.annotator = annotator
        self.detector = detector
        self.tracker = tracker
        self.cache = cache
        self.hooks = hooks or []
        self.privacy = privacy
        self.unblur_until = 0.0  # monotonic deadline of an audited admin unblur (P10)
        self.publish_interval = 1.0 / publish_fps
        self.infer_interval = 1.0 / max_infer_fps
        self.jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        self.mode = "cached" if cache is not None else "realtime"
        self.process_rate = RateMeter()
        self.publish_rate = RateMeter()
        self.infer_ms: deque[float] = deque(maxlen=50)
        self.last_tracks: list[Track] = []
        self._last_publish = 0.0
        self._last_infer = 0.0
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None

    # ------------------------------------------------------------------ processing

    def tracks_for(self, frame: Frame) -> tuple[list[Track], float]:
        if self.cache is not None:
            return self.cache.tracks_at(frame.index), 0.0
        assert self.detector is not None and self.tracker is not None
        t0 = time.perf_counter()
        dets = self.detector.detect([frame.image])[0]
        tracks = self.tracker.update(dets, frame.image)
        return tracks, (time.perf_counter() - t0) * 1000

    def step(self, frame: Frame) -> FrameResult:
        """Process one frame and publish it if the output rate allows. Returns the result."""
        tracks, ms = self.tracks_for(frame)
        if ms:
            self.infer_ms.append(ms)
        now = time.monotonic()
        result = FrameResult(frame, tracks, ms)
        for hook in self.hooks:
            hook(result)
        self.last_tracks = result.tracks
        self.process_rate.tick(now)
        # 0.9: a frame arriving a hair early must not halve the output rate (30 fps in, 15 out)
        if now - self._last_publish >= 0.9 * self.publish_interval:
            self._last_publish = now
            self._publish(result)
        return result

    def _publish(self, result: FrameResult) -> None:
        img = result.frame.image
        if self.privacy is not None and time.monotonic() >= self.unblur_until:
            img = self.privacy.frame(img, result.tracks)  # blur heads of unknown/pending people
        img = self.annotator.draw(img, result.tracks)
        ok, buf = cv2.imencode(".jpg", img, self.jpeg_params)
        if not ok:
            return
        self.publish_rate.tick()
        h, w = img.shape[:2]
        meta = {
            "camera_id": self.camera.id,
            "frame": result.frame.index,
            "ts": result.frame.ts,
            "width": w,
            "height": h,
            "tracks": [
                {
                    "track_id": t.track_id,
                    "global_id": t.global_id,
                    "label": t.label,
                    "role": t.role if t.is_person else None,
                    "conf": round(t.conf, 3),
                    "box": [round(v / s, 4) for v, s in zip(t.xyxy, (w, h, w, h), strict=True)],
                }
                for t in result.tracks
            ],
        }
        self.publish(self.camera.id, buf.tobytes(), meta)

    # ------------------------------------------------------------------ thread

    def stats(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera.id,
            "mode": self.mode,
            "process_fps": round(self.process_rate.rate, 1),
            "publish_fps": round(self.publish_rate.rate, 1),
            "infer_ms": round(sum(self.infer_ms) / len(self.infer_ms), 1)
            if self.infer_ms
            else None,
            "tracks": len(self.last_tracks),
        }

    def run(self) -> None:
        log.info("%s: worker started (%s)", self.camera.id, self.mode)
        while not self._stop.is_set():
            if self.mode == "realtime":
                # Cap inference rate; the latest-frame source drops what we skip.
                wait = self._last_infer + self.infer_interval - time.monotonic()
                if wait > 0:
                    self._stop.wait(wait)
                self._last_infer = time.monotonic()
            frame = self.source.read(timeout=1.0)
            if frame is None:
                continue
            try:
                self.step(frame)
            except Exception:
                log.exception("%s: frame %d failed", self.camera.id, frame.index)
        self.source.close()
        log.info("%s: worker stopped", self.camera.id)

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self.run, name=f"worker-{self.camera.id}", daemon=True
        )
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
