from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import Camera
from app.pipeline.annotate import Annotator
from app.pipeline.cache import TrackCache, TrackCacheWriter
from app.pipeline.engine import BatchingDetector, Engine
from app.pipeline.frames import Detections, Frame, Track
from app.pipeline.reid import ColorHistEncoder
from app.pipeline.sources import FrameSource, LatestFrameSource, VideoFileSource
from app.pipeline.tracker import ByteTracker
from app.pipeline.worker import CameraWorker

LABELS = ("person", "bicycle", "car", "motorcycle", "bus", "truck")


def write_video(path: Path, n: int = 20, fps: float = 30.0, size=(320, 240)) -> Path:
    """Synthetic clip: a white 'person' box moving right by 4 px per frame."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        img = np.zeros((size[1], size[0], 3), np.uint8)
        cv2.rectangle(img, (20 + 4 * i, 60), (60 + 4 * i, 180), (255, 255, 255), -1)
        cv2.putText(img, str(i), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        w.write(img)
    w.release()
    return path


class BrightBoxDetector:
    """Fake detector: one 'person' at the bounding box of bright pixels."""

    labels = LABELS

    def __init__(self) -> None:
        self.calls: list[int] = []

    def detect(self, images):
        self.calls.append(len(images))
        out = []
        for img in images:
            ys, xs = np.where(img[:, :, 0] > 200)
            if len(xs) == 0:
                out.append(Detections.empty(LABELS))
                continue
            box = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], np.float32)
            out.append(
                Detections(box, np.array([0.9], np.float32), np.array([0], np.int32), LABELS)
            )
        return out


def dets(boxes, cls=0, conf=0.9) -> Detections:
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    return Detections(b, np.full(len(b), conf, np.float32), np.full(len(b), cls, np.int32), LABELS)


@pytest.fixture
def tracker_cfg(config):
    return config.settings.pipeline.tracker


# ------------------------------------------------------------------ sources


def test_video_file_source_indices_and_loop(tmp_path):
    video = write_video(tmp_path / "v.mp4", n=5)
    src = VideoFileSource("cam", video, loop=True, paced=False)
    assert src.num_frames == 5
    assert [src.read().index for _ in range(7)] == [0, 1, 2, 3, 4, 0, 1]
    src.close()
    once = VideoFileSource("cam", video, loop=False, paced=False)
    assert len([f for f in iter(once.read, None)]) == 5


def test_video_file_source_paces_at_native_fps(tmp_path):
    src = VideoFileSource("cam", write_video(tmp_path / "v.mp4", n=10, fps=50), paced=True)
    t0 = time.monotonic()
    for _ in range(10):
        src.read()
    assert time.monotonic() - t0 == pytest.approx(9 / 50, abs=0.08)


class CountingSource(FrameSource):
    def __init__(self) -> None:
        self.camera_id, self.fps, self.i = "cam", 100.0, 0

    def read(self, timeout=5.0):
        time.sleep(0.005)
        self.i += 1
        return Frame("cam", self.i, time.time(), np.zeros((4, 4, 3), np.uint8))


def test_latest_frame_source_skips_stale_frames():
    src = LatestFrameSource(CountingSource())
    first = src.read(timeout=1)
    time.sleep(0.1)  # consumer is slow: ~20 frames arrive meanwhile
    second = src.read(timeout=1)
    src.close()
    assert second.index - first.index > 5


# ------------------------------------------------------------------ tracker


def test_bytetrack_keeps_ids_on_smooth_motion(tracker_cfg):
    trk = ByteTracker(tracker_cfg)
    ids = []
    for i in range(30):
        tracks = trk.update(dets([[10 + 3 * i, 10, 50 + 3 * i, 110], [200, 50, 240, 150]]))
        ids.append(sorted(t.track_id for t in tracks))
    assert all(x == ids[-1] for x in ids[2:]) and len(ids[-1]) == 2


def test_bytetrack_separates_people_and_vehicles(tracker_cfg):
    trk = ByteTracker(tracker_cfg)
    for _ in range(3):
        d = Detections(
            np.array([[10, 10, 60, 120], [12, 12, 62, 122]], np.float32),
            np.array([0.9, 0.9], np.float32),
            np.array([0, 2], np.int32),
            LABELS,
        )
        tracks = trk.update(d)
    assert sorted(t.label for t in tracks) == ["car", "person"]
    assert len({t.track_id for t in tracks}) == 2


def test_tracker_drops_tiny_boxes(tracker_cfg):
    trk = ByteTracker(tracker_cfg, min_box_height=24)
    for _ in range(3):
        tracks = trk.update(dets([[0, 0, 10, 10], [0, 0, 30, 60]]))
    assert len(tracks) == 1


# ------------------------------------------------------------------ cache


def test_track_cache_roundtrip(tmp_path):
    w = TrackCacheWriter(LABELS)
    w.add(0, [Track(1, (1, 2, 3, 4), 0.5, "person")])
    w.add(2, [Track(1, (2, 2, 4, 4), 0.6, "person"), Track(2, (5, 5, 9, 9), 0.7, "car")])
    w.save(tmp_path / "c" / "tracks.npz", {"num_frames": 3})
    c = TrackCache(tmp_path / "c" / "tracks.npz")
    assert [t.track_id for t in c.tracks_at(0)] == [1]
    assert c.tracks_at(1) == []
    assert [(t.track_id, t.label) for t in c.tracks_at(2)] == [(1, "person"), (2, "car")]
    assert c.tracks_at(99) == []


# ------------------------------------------------------------------ worker / engine


def camera(video: Path, mode: str = "realtime") -> Camera:
    return Camera(id="cam1", name="Cam", source_type="file", source_uri=str(video), run_mode=mode)


def test_worker_realtime_publishes_normalized_tracks(tmp_path, config, tracker_cfg):
    video = write_video(tmp_path / "v.mp4", n=10)
    published = []
    w = CameraWorker(
        camera(video), VideoFileSource("cam1", video, paced=False),
        lambda cam, jpeg, meta: published.append((cam, jpeg, meta)),
        Annotator(config.settings.pipeline.annotate),
        detector=BrightBoxDetector(), tracker=ByteTracker(tracker_cfg), publish_fps=1e9,
    )  # fmt: skip
    for _ in range(10):
        w.step(w.source.read())
    cam, jpeg, meta = published[-1]
    assert cam == "cam1" and jpeg[:2] == b"\xff\xd8"
    assert meta["width"] == 320 and meta["frame"] == 9
    (t,) = meta["tracks"]
    assert t["label"] == "person" and t["role"] == "pending"
    assert all(0 <= v <= 1 for v in t["box"])
    assert w.stats()["infer_ms"] is not None


def test_worker_cached_mode_uses_cache(tmp_path, config):
    video = write_video(tmp_path / "v.mp4", n=5)
    wr = TrackCacheWriter(LABELS)
    for i in range(5):
        wr.add(i, [Track(7, (10.0 + i, 10, 50, 100), 0.9, "person")])
    wr.save(tmp_path / "t.npz", {"num_frames": 5})
    published = []
    w = CameraWorker(
        camera(video, "cached"), VideoFileSource("cam1", video, paced=False),
        lambda c, j, m: published.append(m), Annotator(config.settings.pipeline.annotate),
        cache=TrackCache(tmp_path / "t.npz"), publish_fps=1e9,
    )  # fmt: skip
    for _ in range(5):
        w.step(w.source.read())
    assert [m["tracks"][0]["track_id"] for m in published] == [7] * 5
    assert w.mode == "cached" and w.stats()["infer_ms"] is None


def test_worker_requires_cache_or_detector(tmp_path, config):
    video = write_video(tmp_path / "v.mp4", n=2)
    with pytest.raises(ValueError):
        CameraWorker(camera(video), VideoFileSource("cam1", video), lambda *a: None,
                     Annotator(config.settings.pipeline.annotate))  # fmt: skip


def test_batching_detector_groups_concurrent_requests():
    inner = BrightBoxDetector()
    bd = BatchingDetector(inner, max_batch=4, wait_ms=50)
    img = np.zeros((10, 10, 3), np.uint8)
    threads = [threading.Thread(target=bd.detect, args=([img],)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    bd.close()
    assert sum(inner.calls) == 4 and max(inner.calls) > 1


def test_engine_runs_realtime_file_camera(tmp_path, config):
    video = write_video(tmp_path / "v.mp4", n=30)
    cfg = config.model_copy(
        update={"cameras": config.cameras.model_copy(update={"cameras": [camera(video)]})}
    )
    got: list[dict] = []
    eng = Engine(
        cfg, lambda c, j, m: got.append(m), detector=BrightBoxDetector(), encoder=ColorHistEncoder()
    )
    eng.start()

    def has_gid() -> bool:
        return any(t["global_id"] == 1 for m in list(got) for t in m["tracks"])

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not (len(got) >= 5 and has_gid()):
        time.sleep(0.05)
    stats = eng.stats()
    eng.stop()
    assert len(got) >= 5
    assert any(m["tracks"] for m in got)
    # the moving box is one person: after a few Re-ID samples it gets global id 1
    assert has_gid()
    assert stats["cameras"][0]["camera_id"] == "cam1"


def test_engine_falls_back_to_realtime_without_cache(tmp_path, config):
    video = write_video(tmp_path / "v.mp4", n=5)
    cfg = config.model_copy(
        update={"cameras": config.cameras.model_copy(update={"cameras": [camera(video, "cached")]})}
    )
    eng = Engine(cfg, lambda *a: None, detector=BrightBoxDetector(), encoder=ColorHistEncoder())
    assert eng.workers[0].mode == "realtime"


def test_capture_source_backs_off_when_stream_is_down(caplog):
    from app.pipeline.sources import CaptureSource

    src = CaptureSource("cam", "rtsp://127.0.0.1:1/none", reconnect_s=0.2)
    t0 = time.monotonic()
    calls = 0
    while time.monotonic() - t0 < 1.0:
        assert src.read(timeout=0.3) is None
        calls += 1
    src.close()
    assert calls <= 6  # waits between attempts instead of spinning
    assert sum("cannot open" in r.message for r in caplog.records) == 1
