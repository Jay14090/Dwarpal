"""Frame sources: video files (looped, paced at native fps), RTSP streams and local webcams.

Realtime cameras are wrapped in LatestFrameSource: a reader thread keeps only the newest frame,
so slow inference skips frames instead of building latency (CLAUDE.md: inference must not
block streaming). Cached cameras read every frame in order so frame indices match the cache.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2

from app.core.config import Camera
from app.pipeline.frames import Frame

log = logging.getLogger(__name__)


class FrameSource(ABC):
    camera_id: str
    fps: float

    @abstractmethod
    def read(self, timeout: float = 5.0) -> Frame | None:
        """Next frame, or None if none arrived within `timeout` (or the source ended)."""

    def close(self) -> None:  # noqa: B027 - optional hook
        pass

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class VideoFileSource(FrameSource):
    """Sequential file reader. `loop` restarts at EOF; `paced` sleeps to play at native fps.

    Frame.index is the position in the file, so cached detections line up after every loop.
    """

    def __init__(
        self,
        camera_id: str,
        path: Path,
        loop: bool = True,
        paced: bool = True,
        epoch: float | None = None,
    ) -> None:
        """`epoch`: if set, Frame.ts = epoch + loops * duration + index / fps, so file cameras
        started together share one timeline (needed for cross-camera timing constraints)."""
        self.camera_id = camera_id
        self.epoch = epoch
        self.loops = 0
        self.path = path
        self.loop = loop
        self.paced = paced
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._index = 0
        self._t0 = time.monotonic()

    def read(self, timeout: float = 5.0) -> Frame | None:
        ok, image = self.cap.read()
        if not ok:
            if not self.loop:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.loops += 1
            self._index = 0
            self._t0 = time.monotonic()
            ok, image = self.cap.read()
            if not ok:
                return None
        if self.paced:
            delay = self._t0 + self._index / self.fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -1.0:  # fell behind by >1 s (overload, debugger): resync the clock
                self._t0 = time.monotonic() - self._index / self.fps
        if self.epoch is None:
            ts = time.time()
        else:
            ts = self.epoch + (self.loops * self.num_frames + self._index) / self.fps
        frame = Frame(self.camera_id, self._index, ts, image)
        self._index += 1
        return frame

    def close(self) -> None:
        self.cap.release()


class CaptureSource(FrameSource):
    """RTSP URL or webcam index read with OpenCV; reconnects with backoff when the stream drops."""

    def __init__(self, camera_id: str, uri: str | int, reconnect_s: float = 2.0) -> None:
        self.camera_id = camera_id
        self.uri = uri
        self.fps = 0.0
        self.reconnect_s = reconnect_s
        self.cap: cv2.VideoCapture | None = None
        self._index = 0
        self._failures = 0
        self._closed = threading.Event()

    def _connect(self) -> bool:
        if isinstance(self.uri, int):
            cap = cv2.VideoCapture(self.uri)
        else:
            cap = cv2.VideoCapture(self.uri, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return False
        self.cap = cap
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        log.info("%s: connected to %s (%.1f fps)", self.camera_id, self.uri, self.fps)
        return True

    def read(self, timeout: float = 5.0) -> Frame | None:
        deadline = time.monotonic() + timeout
        while not self._closed.is_set():
            if self.cap is None:
                if self._connect():
                    self._failures = 0
                else:
                    self._failures += 1
                    if (
                        self._failures == 1 or self._failures % 30 == 0
                    ):  # once per outage, then ~1/min
                        log.warning("%s: cannot open %s (attempt %d), retrying every %.0fs",
                                    self.camera_id, self.uri, self._failures, self.reconnect_s)  # fmt: skip
            if self.cap is not None:
                ok, image = self.cap.read()
                if ok:
                    frame = Frame(self.camera_id, self._index, time.time(), image)
                    self._index += 1
                    return frame
                log.warning("%s: stream dropped, reconnecting", self.camera_id)
                self.cap.release()
                self.cap = None
            # back off before the next attempt, but never past the caller's deadline
            self._closed.wait(min(self.reconnect_s, max(0.0, deadline - time.monotonic())))
            if time.monotonic() >= deadline:
                return None
        return None

    def close(self) -> None:
        self._closed.set()
        if self.cap is not None:
            self.cap.release()


class LatestFrameSource(FrameSource):
    """Runs `inner.read()` in a background thread and hands out only the newest frame."""

    def __init__(self, inner: FrameSource) -> None:
        self.inner = inner
        self.camera_id = inner.camera_id
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._delivered: Frame | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"src-{self.camera_id}", daemon=True)
        self._thread.start()

    @property
    def fps(self) -> float:  # type: ignore[override]
        return self.inner.fps

    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self.inner.read(timeout=1.0)
            if frame is None:
                continue
            with self._cond:
                self._latest = frame
                self._cond.notify_all()

    def read(self, timeout: float = 5.0) -> Frame | None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._latest is None or self._latest is self._delivered:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stop.is_set():
                    return None
                self._cond.wait(remaining)
            self._delivered = self._latest
            return self._latest

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=3)
        self.inner.close()


def open_source(camera: Camera, root: Path, epoch: float | None = None) -> FrameSource:
    if camera.source_type == "file":
        path = Path(camera.source_uri)
        src: FrameSource = VideoFileSource(
            camera.id, path if path.is_absolute() else root / path, epoch=epoch
        )
        return src if camera.run_mode == "cached" else LatestFrameSource(src)
    uri: str | int = camera.source_uri if camera.source_type == "rtsp" else int(camera.source_uri)
    return LatestFrameSource(CaptureSource(camera.id, uri))
