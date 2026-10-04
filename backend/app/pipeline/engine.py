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
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.anpr.hook import PlateHook, PlateReadCache, plates_cache_path
from app.anpr.ocr import OcrRead, PlateOcr, build_plate_ocr
from app.anpr.plate_detector import PlateBox, PlateDetector, build_plate_detector
from app.anpr.registry import Registry
from app.anpr.service import PlateService
from app.core.config import Camera, Config, load_config
from app.core.device import resolve_device
from app.core.logging import setup_logging
from app.datasets.common import Calibration
from app.db.models import Event as DbEvent
from app.db.track_store import TrackStore, upsert_identity
from app.events import EventBus, EventRecord
from app.pipeline.annotate import Annotator
from app.pipeline.cache import TrackCache, cache_path
from app.pipeline.clip import ClipEncoder
from app.pipeline.crosscam import CrossCameraHook, ReidCache, reid_cache_path
from app.pipeline.detector import Detector, YoloDetector
from app.pipeline.face import FaceEncoder, FaceSample, build_face_encoder
from app.pipeline.frames import Detections
from app.pipeline.global_tracker import GlobalTracker
from app.pipeline.identity import Gallery, IdentityEngine
from app.pipeline.indexer import DbWriter, FinishedTrack, ImageEncoder, TrackIndexer
from app.pipeline.person_hooks import (
    EnrollmentCollector,
    FaceHook,
    IdentityHook,
    SampleCache,
    face_cache_path,
)
from app.pipeline.reid import ReidEncoder, build_encoder
from app.pipeline.sources import open_source
from app.pipeline.tracker import ByteTracker
from app.pipeline.worker import CameraWorker, FrameHook, PublishFn
from app.rules import RulesEngine, RulesHook

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
        face_encoder: FaceEncoder | None = None,
        identity: IdentityEngine | None = None,
        gallery_loader: Callable[[], Gallery] | None = None,
        plate_detector: PlateDetector | None = None,
        plate_ocr: PlateOcr | None = None,
        registry_loader: Callable[[], Registry] | None = None,
        db_writer: DbWriter | None = None,
        emit: Callable[[str, dict[str, Any]], None] | None = None,
        clip_encoder: ImageEncoder | None = None,
        track_sink: Callable[[FinishedTrack], None] | None = None,
        global_start_id: int = 1,
    ) -> None:
        self.config = config
        s = config.settings
        self.emit = emit
        self.db_writer = db_writer
        self.event_bus = EventBus(config.rules, self._event_sink)
        self.rules_engine = RulesEngine(
            config.rules, self.event_bus, s.app.timezone, config.rules.defaults.presence_gap_s
        )
        self._plate_det = plate_detector
        self._plate_ocr = plate_ocr
        self._anpr_error: Exception | None = None
        self.registry_loader = registry_loader
        self.plate_service = PlateService(Registry(max_distance=s.anpr.registry_max_distance),
                                          s.paths.thumbs_dir, self.event_bus, db_writer, emit)  # fmt: skip
        self.reload_registry()
        cams = [c for c in config.cameras.cameras if c.enabled]
        if camera_ids:
            cams = [c for c in config.cameras.cameras if c.id in camera_ids]
        self.annotator = Annotator(s.pipeline.annotate)
        self._detector = detector
        self._encoder: ReidEncoder | None = LockedEncoder(encoder) if encoder else None
        self._encoder_error: Exception | None = None
        self._face: FaceEncoder | None = LockedFaceEncoder(face_encoder) if face_encoder else None
        self._face_error: Exception | None = (
            None if s.face.enabled else RuntimeError("face.enabled=false")
        )
        self.global_tracker = global_tracker or GlobalTracker(
            s.global_tracker, s.reid.max_samples, global_start_id
        )
        self.identity = identity or IdentityEngine(s.identity)
        self._clip: ImageEncoder | None = LockedClip(clip_encoder) if clip_encoder else None
        self._clip_error: Exception | None = None
        self.track_store = (
            TrackStore(db_writer, s.paths.thumbs_dir) if db_writer is not None else None
        )
        self.track_sink = track_sink or (self.track_store.store if self.track_store else None)
        self.indexers: list[TrackIndexer] = []
        if self.track_store is not None:
            store = self.track_store
            self.global_tracker.new_identity_listeners.append(
                lambda ident: store.identity_created(ident.id, ident.first_seen)
            )
            self.identity.listeners.append(
                lambda st, prev, ts: store.identity_changed(
                    st.global_id, st.role_state, st.person_id, ts
                )
            )
        self.gallery_loader = gallery_loader
        self.reload_gallery()
        self.collectors: list[tuple[CameraWorker, EnrollmentCollector]] = []
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

    # ------------------------------------------------------------------ hooks

    def hooks_for(self, cam: Camera, cached: bool, fps: float) -> list[FrameHook]:
        """CrossCameraHook (global ids) -> FaceHook -> IdentityHook (roles)."""
        s = self.config.settings
        reid_cache = face_cache = None
        if cached:
            path = reid_cache_path(s.paths.cache_dir, cam.id)
            if path.is_file():
                reid_cache = ReidCache(path)
            else:
                log.warning("%s: no reid cache at %s; computing embeddings live", cam.id, path)
            fpath = face_cache_path(s.paths.cache_dir, cam.id)
            if fpath.is_file():
                face_cache = SampleCache(fpath)
        encoder = None
        if reid_cache is None:
            encoder = self.encoder
            if encoder is None:
                log.error("%s: Re-ID unavailable (%s); no global IDs", cam.id, self._encoder_error)
                return (self.plate_hooks(cam, cached, fps) if cam.anpr else []) + self.index_hooks(
                    cam, fps
                )
        # A local track lost for longer than the tracker keeps it is over.
        grace_s = s.pipeline.tracker.track_buffer / max(fps, 1.0) + 0.5
        hooks: list[FrameHook] = [
            CrossCameraHook(
                cam.id, s.reid, self.global_tracker, encoder=encoder, cache=reid_cache,
                calibration=load_calibration_for(cam, self.config), grace_s=grace_s,
            )
        ]  # fmt: skip
        if face_cache is not None:
            hooks.append(FaceHook(cam.id, s.face, cache=face_cache))
        elif not cached and self.face_encoder is not None:
            hooks.append(FaceHook(cam.id, s.face, encoder=self.face_encoder))
        hooks.append(IdentityHook(self.identity))
        hooks.append(RulesHook(cam, self.rules_engine))
        if cam.anpr:
            hooks.extend(self.plate_hooks(cam, cached, fps))
        hooks.extend(self.index_hooks(cam, fps))
        return hooks

    def index_hooks(self, cam: Camera, fps: float) -> list[FrameHook]:
        if self.track_sink is None:
            return []
        s = self.config.settings
        indexer = TrackIndexer(
            cam, s.index, self.track_sink, clip=self.clip_encoder, clip_top_k=s.clip.top_k,
            calibration=load_calibration_for(cam, self.config),
            grace_frames=int(s.pipeline.tracker.track_buffer + fps),
        )  # fmt: skip
        self.indexers.append(indexer)
        return [indexer]

    @property
    def clip_encoder(self) -> ImageEncoder | None:
        if self._clip is None and self._clip_error is None:
            s = self.config.settings
            try:
                self._clip = LockedClip(
                    ClipEncoder(s.clip, resolve_device(s.device), s.half_precision)
                )
            except Exception as exc:
                log.error(
                    "CLIP unavailable (%s); tracks are indexed without search embeddings", exc
                )
                self._clip_error = exc
        return self._clip

    def plate_hooks(self, cam: Camera, cached: bool, fps: float) -> list[FrameHook]:
        s = self.config.settings
        grace = int(s.pipeline.tracker.track_buffer + fps)  # tracker keeps lost tracks this long
        path = plates_cache_path(s.paths.cache_dir, cam.id)
        if cached and path.is_file():
            return [
                PlateHook(
                    cam.id,
                    s.anpr,
                    self.plate_service.handle,
                    cache=PlateReadCache(path),
                    grace_frames=grace,
                )
            ]
        if self.plate_models is None:
            log.error("%s: ANPR unavailable (%s)", cam.id, self._anpr_error)
            return []
        det, ocr = self.plate_models
        return [
            PlateHook(
                cam.id, s.anpr, self.plate_service.handle, detector=det, ocr=ocr, grace_frames=grace
            )
        ]

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
    def face_encoder(self) -> FaceEncoder | None:
        if self._face is None and self._face_error is None:
            s = self.config.settings
            try:
                enc = build_face_encoder(s.face, resolve_device(s.device))
                self._face = LockedFaceEncoder(enc) if enc else None
            except Exception as exc:
                log.error("faces unavailable: %s", exc)
                self._face_error = exc
        return self._face

    @property
    def plate_models(self) -> tuple[PlateDetector, PlateOcr] | None:
        if (self._plate_det is None or self._plate_ocr is None) and self._anpr_error is None:
            s = self.config.settings
            try:
                device = resolve_device(s.device)
                self._plate_det = self._plate_det or LockedPlateDetector(
                    build_plate_detector(s.anpr.detector, device)
                )
                self._plate_ocr = self._plate_ocr or LockedPlateOcr(
                    build_plate_ocr(s.anpr.ocr, device)
                )
            except Exception as exc:
                self._anpr_error = exc
        if self._plate_det is None or self._plate_ocr is None:
            return None
        return self._plate_det, self._plate_ocr

    def _event_sink(self, ev: EventRecord) -> None:
        log.info("event %s (%s) on %s: %s", ev.rule, ev.severity, ev.camera_id, ev.payload)

        def done(event_id: int | None) -> None:
            ev.id = event_id
            if self.emit is not None:
                self.emit("event", ev.as_dict())

        if self.db_writer is None:
            done(None)
        else:
            thumbs = self.config.settings.paths.thumbs_dir
            self.db_writer.submit(lambda session: store_event(session, ev, thumbs), done)

    def reload_registry(self) -> int:
        if self.registry_loader is None:
            return len(self.plate_service.registry.plates)
        try:
            reg = self.registry_loader()
        except Exception as exc:
            log.error("vehicle registry load failed (%s)", exc)
            return len(self.plate_service.registry.plates)
        self.plate_service.set_registry(reg)
        log.info("vehicle registry: %d plates", len(reg.plates))
        return len(reg.plates)

    @property
    def detector(self) -> Detector:
        """Created on first use so all-cached setups never load the model."""
        if self._detector is None:
            s = self.config.settings
            device = resolve_device(s.device)
            yolo = YoloDetector(s.pipeline.detector, device, half=s.half_precision)
            self._detector = BatchingDetector(yolo, max_batch=s.pipeline.detector.batch_size)
        return self._detector

    # ------------------------------------------------------------------ control

    def reload_gallery(self) -> int:
        if self.gallery_loader is None:
            return len(self.identity.gallery)
        try:
            gallery = self.gallery_loader()
        except Exception as exc:
            log.error(
                "gallery load failed (%s); keeping %d people", exc, len(self.identity.gallery)
            )
            return len(self.identity.gallery)
        self.identity.set_gallery(gallery, time.time())
        return len(gallery)

    def start_capture(
        self, camera_id: str, seconds: float, shots: int, on_done: Callable[[dict[str, Any]], None]
    ) -> None:
        worker = next((w for w in self.workers if w.camera.id == camera_id), None)
        if worker is None:
            on_done({"error": f"camera {camera_id} is not running"})
            return
        collector = EnrollmentCollector(seconds, shots, self.face_encoder, self.encoder, on_done)
        worker.hooks = [*worker.hooks, collector]  # copy-on-write: the worker thread iterates
        self.collectors.append((worker, collector))

    def tick(self) -> None:
        """Periodic housekeeping from the engine main loop."""
        for worker, c in list(self.collectors):
            c.expire()
            if c.done:
                worker.hooks = [h for h in worker.hooks if h is not c]
                self.collectors.remove((worker, c))

    def embed_images(self, images: list[np.ndarray]) -> dict[str, Any]:
        """Embeddings for uploaded enrollment photos: best face per photo (+ body if the photo
        looks like a full-body shot)."""
        faces, fq, bodies, bq = [], [], [], []
        for img in images:
            h, w = img.shape[:2]
            if self.face_encoder is not None:
                fs = self.face_encoder.faces_for(img, [(0.0, 0.0, float(w), float(h) / 0.45)])[0]
                if fs is not None:
                    faces.append(fs.emb.tolist())
                    fq.append(round(fs.quality, 3))
            if h / max(w, 1) >= 1.6 and self.encoder is not None:
                bodies.append(self.encoder.embed([img])[0].tolist())
                bq.append(1.0)
        return {"face": faces, "face_quality": fq, "body": bodies, "body_quality": bq}

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"cameras": [w.stats() for w in self.workers], "ts": time.time()}
        if isinstance(self._detector, BatchingDetector) and self._detector.batches:
            out["detector_avg_batch"] = round(self._detector.images / self._detector.batches, 2)
        roles: dict[str, int] = {}
        for st in self.identity.states.values():
            roles[st.role_state] = roles.get(st.role_state, 0) + 1
        out["identities"] = {"global_ids": len(self.global_tracker.identities), "roles": roles,
                             "gallery_people": len(self.identity.gallery)}  # fmt: skip
        out["anpr"] = {"plates_decided": len(self.plate_service.recent),
                       "registry": len(self.plate_service.registry.plates)}  # fmt: skip
        if self.db_writer is not None:
            out["db"] = {"writes": self.db_writer.done, "failures": self.db_writer.failures}
        return out

    def start(self) -> None:
        for w in self.workers:
            w.start()

    def stop(self) -> None:
        for w in self.workers:
            w.stop()
        for ix in self.indexers:
            ix.flush()  # tracks still open when the engine stops are indexed too
        if isinstance(self._detector, BatchingDetector):
            self._detector.close()
        if self.db_writer is not None:
            self.db_writer.close()


class LockedFaceEncoder:
    def __init__(self, inner: FaceEncoder) -> None:
        self.inner = inner
        self._lock = threading.Lock()

    def faces_for(
        self, image: np.ndarray, boxes: list[tuple[float, float, float, float]]
    ) -> list[FaceSample | None]:
        with self._lock:
            return self.inner.faces_for(image, boxes)


class LockedClip:
    def __init__(self, inner: ImageEncoder) -> None:
        self.inner = inner
        self._lock = threading.Lock()

    def embed_images(self, crops: list[np.ndarray]) -> np.ndarray:
        with self._lock:
            return self.inner.embed_images(crops)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            return self.inner.embed_text(texts)  # type: ignore[attr-defined]


class LockedPlateDetector:
    def __init__(self, inner: PlateDetector) -> None:
        self.inner = inner
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray) -> list[PlateBox]:
        with self._lock:
            return self.inner.detect(image)


class LockedPlateOcr:
    def __init__(self, inner: PlateOcr) -> None:
        self.inner = inner
        self._lock = threading.Lock()

    def read(self, plates: list[np.ndarray]) -> list[OcrRead]:
        with self._lock:
            return self.inner.read(plates)


def store_event(session: Session, ev: EventRecord, thumbs_dir: Path | None = None) -> int:
    if ev.global_id is not None:  # the identity row normally exists already (TrackStore)
        upsert_identity(session, ev.global_id, ev.ts)
    row = DbEvent(
        rule=ev.rule,
        severity=ev.severity,
        camera_id=ev.camera_id,
        ts=datetime.fromtimestamp(ev.ts, UTC),
        global_id=ev.global_id,
        plate_read_id=ev.plate_read_id,
        payload={**ev.payload, "rule_type": ev.rule_type, "global_id": ev.global_id},
    )
    session.add(row)
    session.flush()
    if ev.thumb_jpeg is not None and thumbs_dir is not None:
        path = event_thumb_path(thumbs_dir, row.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(ev.thumb_jpeg)
        row.payload = {**row.payload, "thumb": True}
    return row.id


def event_thumb_path(thumbs_dir: Path, event_id: int) -> Path:
    return thumbs_dir / "events" / f"{event_id}.jpg"


def db_registry_loader(config: Config) -> Callable[[], Registry]:
    from app.anpr.service import load_registry
    from app.db.session import make_engine

    db = make_engine(config.settings.database)

    def load() -> Registry:
        with Session(db) as session:
            return load_registry(session, config.settings.anpr.registry_max_distance)

    return load


def db_gallery_loader(config: Config) -> Callable[[], Gallery]:
    from app.db.gallery_store import load_gallery
    from app.db.session import make_engine

    db = make_engine(config.settings.database)
    version = 0

    def load() -> Gallery:
        nonlocal version
        version += 1
        with Session(db) as session:
            return load_gallery(session, version)

    return load


# --------------------------------------------------------------------------- process wrapper


def engine_main(
    config_dir: str,
    camera_ids: list[str] | None,
    out_q: mp.Queue,
    stop_evt: Any,
    ctrl_q: mp.Queue | None = None,
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

    def reply(req_id: str, payload: dict[str, Any]) -> None:
        out_q.put(("result", req_id, payload), timeout=10)  # results must not be dropped

    def emit(kind: str, payload: dict[str, Any]) -> None:
        try:
            out_q.put((kind, payload), timeout=5)  # events/plates: block briefly rather than drop
        except queue.Full:
            log.error("API not draining; dropped %s", kind)

    from app.db.session import make_engine
    from app.db.sync import sync_cameras
    from app.db.track_store import next_global_id

    db = make_engine(config.settings.database)
    writer = DbWriter(db)
    writer.submit(lambda session: sync_cameras(session, config.cameras))
    try:
        with Session(db) as session:
            start_id = next_global_id(session)
    except Exception:
        start_id = 1
    engine = Engine(
        config, publish, camera_ids,
        gallery_loader=db_gallery_loader(config), registry_loader=db_registry_loader(config),
        db_writer=writer, emit=emit, global_start_id=start_id,
    )  # fmt: skip
    engine.start()
    next_stats = time.monotonic()
    try:
        while not stop_evt.is_set():
            try:
                msg = ctrl_q.get(timeout=0.25) if ctrl_q is not None else stop_evt.wait(0.25)
            except queue.Empty:
                msg = None
            if isinstance(msg, tuple):
                handle_command(engine, msg, reply)
            engine.tick()
            if time.monotonic() >= next_stats:
                next_stats = time.monotonic() + stats_every_s
                stats = engine.stats()
                stats["dropped_frames"] = dropped
                with contextlib.suppress(queue.Full):
                    out_q.put_nowait(("stats", stats))
    finally:
        engine.stop()


def handle_command(
    engine: Engine, msg: tuple, reply: Callable[[str, dict[str, Any]], None]
) -> None:
    cmd, req_id, args = msg
    try:
        if cmd == "reload_gallery":
            reply(req_id, {"people": engine.reload_gallery()})
        elif cmd == "reload_registry":
            reply(req_id, {"plates": engine.reload_registry()})
        elif cmd == "enroll_capture":
            engine.start_capture(
                args["camera_id"], args["seconds"], args["shots"], lambda r: reply(req_id, r)
            )
        elif cmd == "embed_images":
            imgs = [
                cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in args["images"]
            ]
            reply(req_id, engine.embed_images([i for i in imgs if i is not None]))
        elif cmd == "embed_text":
            clip = engine.clip_encoder
            if clip is None:
                reply(req_id, {"error": "CLIP is not available"})
            else:
                reply(req_id, {"vectors": clip.embed_text(args["texts"]).tolist()})  # type: ignore[attr-defined]
        else:
            reply(req_id, {"error": f"unknown command {cmd}"})
    except Exception as exc:
        log.exception("command %s failed", cmd)
        reply(req_id, {"error": str(exc)})


class FrameHub:
    """API-side store of the latest JPEG + metadata per camera (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[str, tuple[int, float, bytes, dict[str, Any]]] = {}
        self.stats: dict[str, Any] = {}
        self.events: deque[dict[str, Any]] = deque(maxlen=500)
        self.plates: deque[dict[str, Any]] = deque(maxlen=500)
        self.listeners: list[Callable[[str, dict[str, Any]], None]] = []  # WebSocket fan-out (P8)

    def push(self, kind: str, payload: dict[str, Any]) -> None:
        (self.events if kind == "event" else self.plates).appendleft(payload)
        for cb in list(self.listeners):
            cb(kind, payload)

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
        self._ctrl: mp.Queue = ctx.Queue()
        self._stop = ctx.Event()
        self._pending: dict[str, tuple[threading.Event, list[dict[str, Any]]]] = {}
        self._proc = ctx.Process(
            target=engine_main,
            args=(str(config.config_dir), camera_ids, self._q, self._stop, self._ctrl),
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
            elif msg[0] in ("event", "plate"):
                self.hub.push(msg[0], msg[1])
            elif msg[0] == "result":
                pending = self._pending.get(msg[1])
                if pending is not None:
                    pending[1].append(msg[2])
                    pending[0].set()

    def request(self, cmd: str, timeout: float, **args: Any) -> dict[str, Any]:
        """Send a command to the engine and wait for its reply (blocking; call from a thread)."""
        req_id = uuid.uuid4().hex
        done = threading.Event()
        box: list[dict[str, Any]] = []
        self._pending[req_id] = (done, box)
        try:
            self._ctrl.put((cmd, req_id, args))
            if not done.wait(timeout):
                raise TimeoutError(f"engine did not answer {cmd} within {timeout:.0f}s")
            return box[0]
        finally:
            self._pending.pop(req_id, None)

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
