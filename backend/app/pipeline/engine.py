"""Engine: runs all camera workers in one process, separate from the API process.

API process                          engine process (spawned)
  EngineProcess.start()  ----------->  engine_main(): Engine(config).run()
  drain thread  <-- mp.Queue ---------   workers publish ("frame", cam, jpeg, meta)
  FrameHub (latest JPEG per camera)      and periodic ("stats", {...})
  MJPEG endpoints read the hub

One BatchingDetector is shared by all realtime cameras: requests that arrive within a few ms
are run as one batch, which is how several cameras fit on one small GPU.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any

import numpy as np

from app.core.config import Camera, Config, load_config
from app.core.device import resolve_device
from app.core.logging import setup_logging
from app.datasets.common import Calibration
from app.pipeline.annotate import Annotator
from app.pipeline.cache import TrackCache, cache_path
from app.pipeline.crosscam import CrossCameraHook, ReidCache, reid_cache_path
from app.pipeline.detector import Detector, YoloDetector
from app.pipeline.frames import Detections
from app.pipeline.global_tracker import GlobalTracker
from app.pipeline.reid import ReidEncoder, build_encoder
from app.pipeline.sources import open_source
from app.pipeline.tracker import ByteTracker
from app.pipeline.worker import CameraWorker, FrameHook, PublishFn

log = logging.getLogger(__name__)


class BatchingDetector:
    """Thread-safe front for a Detector that batches concurrent single-image requests."""

    def __init__(self, inner: Detector, max_batch: int = 4, wait_ms: float = 4.0) -> None:
        self.inner = inner
        self.labels = inner.labels
        self.max_batch = max_batch
        self.wait_s = wait_ms / 1000
        self._q: queue.Queue[tuple[np.ndarray, Future[Detections]]] = queue.Queue()
        self._stop = threading.Event()
        self.batches = 0
        self.images = 0
        self._thread = threading.Thread(target=self._run, name="batch-detector", daemon=True)
        self._thread.start()

    def detect(self, images: list[np.ndarray]) -> list[Detections]:
        futures: list[Future[Detections]] = []
        for img in images:
            fut: Future[Detections] = Future()
            self._q.put((img, fut))
            futures.append(fut)
        return [f.result() for f in futures]

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + self.wait_s
            while len(batch) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._q.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                results = self.inner.detect([img for img, _ in batch])
                for (_, fut), res in zip(batch, results, strict=True):
                    fut.set_result(res)
                self.batches += 1
                self.images += len(batch)
            except Exception as exc:
                for _, fut in batch:
                    fut.set_exception(exc)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


class LockedEncoder:
    """Serializes Re-ID calls from several camera threads onto one model."""

    def __init__(self, inner: ReidEncoder) -> None:
        self.inner = inner
        self.dim = inner.dim
        self._lock = threading.Lock()

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        with self._lock:
            return self.inner.embed(crops)


def load_calibration_for(camera: Camera, config: Config) -> Calibration | None:
    """Dataset calibration (processed/<dataset>/calibration/<id>.json), else a manual one in
    cameras.yaml (`calibration: {P: 3x4, width, height}`), else None (no ground plane)."""
    if camera.dataset:
        path = (
            config.settings.paths.processed_dir
            / camera.dataset
            / "calibration"
            / f"{camera.id}.json"
        )
        if path.is_file():
            return Calibration.load(path)
    if camera.calibration and "P" in camera.calibration:
        return Calibration.from_json({"camera": camera.id, **camera.calibration})
    return None


class Engine:
    def __init__(
        self,
        config: Config,
        publish: PublishFn,
        camera_ids: list[str] | None = None,
        detector: Detector | None = None,
        encoder: ReidEncoder | None = None,
        global_tracker: GlobalTracker | None = None,
    ) -> None:
        self.config = config
        s = config.settings
        cams = [c for c in config.cameras.cameras if c.enabled]
        if camera_ids:
            cams = [c for c in config.cameras.cameras if c.id in camera_ids]
        self.annotator = Annotator(s.pipeline.annotate)
        self._detector = detector
        self._encoder: ReidEncoder | None = LockedEncoder(encoder) if encoder else None
        self._encoder_error: Exception | None = None
        self.global_tracker = global_tracker or GlobalTracker(s.global_tracker, s.reid.max_samples)
        self.epoch = time.time()  # shared timeline for all file cameras
        self.workers: list[CameraWorker] = []
        for cam in cams:
            cache = None
            if cam.run_mode == "cached":
                path = cache_path(s.paths.cache_dir, cam.id)
                if path.is_file():
                    cache = TrackCache(path)
                else:
                    log.warning(
                        "%s: no cache at %s (run `make index`); running realtime", cam.id, path
                    )
            try:
                source = open_source(cam, config.root_dir, epoch=self.epoch)
            except FileNotFoundError as exc:
                log.error("%s: %s; camera skipped", cam.id, exc)
                continue
            n_video = getattr(source, "num_frames", 0)
            if cache is not None and n_video not in (0, cache.num_frames):
                log.warning("%s: cache has %d frames, video %d", cam.id, cache.num_frames, n_video)
            kwargs: dict[str, Any] = {"cache": cache}
            if cache is None:
                kwargs["detector"] = self.detector
                kwargs["tracker"] = ByteTracker(s.pipeline.tracker, s.pipeline.min_box_height_px)
            self.workers.append(
                CameraWorker(
                    cam, source, publish, self.annotator,
                    hooks=self.hooks_for(cam, cached=cache is not None, fps=source.fps or 30.0),
                    publish_fps=s.streaming.mjpeg_max_fps,
                    jpeg_quality=s.streaming.mjpeg_jpeg_quality,
                    max_infer_fps=s.pipeline.realtime_max_fps,
                    **kwargs,
                )
            )  # fmt: skip

    def hooks_for(self, cam: Camera, cached: bool, fps: float) -> list[FrameHook]:
        s = self.config.settings
        reid_cache = None
        if cached:
            path = reid_cache_path(s.paths.cache_dir, cam.id)
            if path.is_file():
                reid_cache = ReidCache(path)
            else:
                log.warning("%s: no reid cache at %s; computing embeddings live", cam.id, path)
        encoder = None
        if reid_cache is None:
            encoder = self.encoder
            if encoder is None:
                log.error("%s: Re-ID unavailable (%s); no global IDs", cam.id, self._encoder_error)
                return []
        # A local track lost for longer than the tracker keeps it is over.
        grace_s = s.pipeline.tracker.track_buffer / max(fps, 1.0) + 0.5
        return [
            CrossCameraHook(
                cam.id, s.reid, self.global_tracker, encoder=encoder, cache=reid_cache,
                calibration=load_calibration_for(cam, self.config), grace_s=grace_s,
            )
        ]  # fmt: skip

    @property
    def encoder(self) -> ReidEncoder | None:
        """Built on first use; None (with the reason kept) if weights cannot be loaded."""
        if self._encoder is None and self._encoder_error is None:
            s = self.config.settings
            try:
                self._encoder = LockedEncoder(
                    build_encoder(s.reid, resolve_device(s.device), s.half_precision)
                )
            except Exception as exc:
                self._encoder_error = exc
        return self._encoder

    @property
    def detector(self) -> Detector:
        """Created on first use so all-cached setups never load the model."""
        if self._detector is None:
            s = self.config.settings
            device = resolve_device(s.device)
            yolo = YoloDetector(s.pipeline.detector, device, half=s.half_precision)
            self._detector = BatchingDetector(yolo, max_batch=s.pipeline.detector.batch_size)
        return self._detector

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"cameras": [w.stats() for w in self.workers], "ts": time.time()}
        if isinstance(self._detector, BatchingDetector) and self._detector.batches:
            out["detector_avg_batch"] = round(self._detector.images / self._detector.batches, 2)
        return out

    def start(self) -> None:
        for w in self.workers:
            w.start()

    def stop(self) -> None:
        for w in self.workers:
            w.stop()
        if isinstance(self._detector, BatchingDetector):
            self._detector.close()


# --------------------------------------------------------------------------- process wrapper


def engine_main(
    config_dir: str,
    camera_ids: list[str] | None,
    out_q: mp.Queue,
    stop_evt: Any,
    stats_every_s: float = 2.0,
) -> None:
    """Entry point of the engine child process."""
    config = load_config(config_dir)
    setup_logging(config.settings.app.log_level)
    dropped = 0

    def publish(camera_id: str, jpeg: bytes, meta: dict[str, Any]) -> None:
        nonlocal dropped
        try:
            out_q.put_nowait(("frame", camera_id, jpeg, meta))
        except queue.Full:
            dropped += 1  # API side is slow: drop frames, never block inference

    engine = Engine(config, publish, camera_ids)
    engine.start()
    try:
        while not stop_evt.wait(stats_every_s):
            stats = engine.stats()
            stats["dropped_frames"] = dropped
            with contextlib.suppress(queue.Full):
                out_q.put_nowait(("stats", stats))
    finally:
        engine.stop()


class FrameHub:
    """API-side store of the latest JPEG + metadata per camera (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[str, tuple[int, float, bytes, dict[str, Any]]] = {}
        self.stats: dict[str, Any] = {}

    def put(self, camera_id: str, jpeg: bytes, meta: dict[str, Any]) -> None:
        with self._lock:
            version = self._frames.get(camera_id, (0,))[0] + 1
            self._frames[camera_id] = (version, time.time(), jpeg, meta)

    def get(self, camera_id: str) -> tuple[int, float, bytes, dict[str, Any]] | None:
        with self._lock:
            return self._frames.get(camera_id)

    def cameras(self) -> list[str]:
        with self._lock:
            return list(self._frames)


class EngineProcess:
    """Starts the engine in a spawned child process and feeds its output into a FrameHub."""

    def __init__(self, config: Config, hub: FrameHub, camera_ids: list[str] | None = None) -> None:
        self.config = config
        self.hub = hub
        self.camera_ids = camera_ids
        ctx = mp.get_context("spawn")  # CUDA-safe
        self._q: mp.Queue = ctx.Queue(maxsize=config.settings.pipeline.frame_queue_size)
        self._stop = ctx.Event()
        self._proc = ctx.Process(
            target=engine_main,
            args=(str(config.config_dir), camera_ids, self._q, self._stop),
            name="dwarpal-engine",
            daemon=True,
        )
        self._drain_stop = threading.Event()
        self._drain = threading.Thread(target=self._drain_loop, name="engine-drain", daemon=True)

    def _drain_loop(self) -> None:
        while not self._drain_stop.is_set():
            try:
                msg = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                break
            if msg[0] == "frame":
                _, cam, jpeg, meta = msg
                self.hub.put(cam, jpeg, meta)
            elif msg[0] == "stats":
                self.hub.stats = msg[1]

    @property
    def alive(self) -> bool:
        return self._proc.is_alive()

    def start(self) -> None:
        self._proc.start()
        self._drain.start()
        log.info("engine process started (pid %s)", self._proc.pid)

    def stop(self) -> None:
        self._stop.set()
        self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.terminate()
        self._drain_stop.set()
        self._drain.join(timeout=2)
