"""Precompute tracked detections for cached-mode cameras (`make index`).

Detection runs in batches of frames; tracking then consumes them strictly in order. Person
tracks are sampled for Re-ID with the same CrossCameraHook the live engine uses, so cached
replay sees exactly the samples realtime mode would take.
Writes data/cache/<camera>/tracks.npz and reid.npz and prints throughput.

Usage: python scripts/index_cameras.py [camera ...] [--weights models/yolo26n.pt]
       [--max-frames N] [--no-reid] [--reid-backend colorhist]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from app.core.config import FaceSettings, ReidSettings, get_config
from app.core.device import resolve_device
from app.pipeline.cache import TrackCacheWriter, cache_path
from app.pipeline.crosscam import CrossCameraHook, ReidCacheWriter, reid_cache_path
from app.pipeline.detector import YoloDetector
from app.pipeline.face import FaceEncoder, build_face_encoder
from app.pipeline.frames import FrameResult
from app.pipeline.person_hooks import FaceHook, face_cache_path
from app.pipeline.reid import ReidEncoder, build_encoder
from app.pipeline.sources import VideoFileSource
from app.pipeline.tracker import ByteTracker

log = logging.getLogger("index")


def index_camera(camera_id: str, video: Path, detector: YoloDetector, out: Path,
                 batch: int, max_frames: int | None, tracker: ByteTracker,
                 encoder: ReidEncoder | None = None, reid_cfg: ReidSettings | None = None,
                 face_encoder: FaceEncoder | None = None, face_cfg: FaceSettings | None = None) -> dict:  # fmt: skip
    src = VideoFileSource(camera_id, video, loop=False, paced=False)
    total = min(src.num_frames, max_frames) if max_frames else src.num_frames
    writer = TrackCacheWriter(detector.labels)
    reid_writer = ReidCacheWriter()
    hook = None
    if encoder is not None and reid_cfg is not None:
        hook = CrossCameraHook(camera_id, reid_cfg, None, encoder=encoder, recorder=reid_writer)
    face_writer = ReidCacheWriter()  # same (frame, track, quality, emb) layout
    face_hook = None
    if face_encoder is not None and face_cfg is not None:
        face_hook = FaceHook(camera_id, face_cfg, encoder=face_encoder, recorder=face_writer)
    t0 = time.perf_counter()
    done = 0
    infer_s = 0.0
    while done < total:
        frames = []
        while len(frames) < batch and done + len(frames) < total:
            f = src.read()
            if f is None:
                break
            frames.append(f)
        if not frames:
            break
        ti = time.perf_counter()
        dets = detector.detect([f.image for f in frames])
        infer_s += time.perf_counter() - ti
        for f, d in zip(frames, dets, strict=True):
            tracks = tracker.update(d, f.image)
            writer.add(f.index, tracks)
            result = FrameResult(f, tracks)
            if hook is not None:
                hook(result)
            if face_hook is not None:
                face_hook(result)
        done += len(frames)
        if done % (batch * 50) < batch:
            el = time.perf_counter() - t0
            log.info("%s: %d/%d frames (%.1f fps)", camera_id, done, total, done / el)
    src.close()
    elapsed = time.perf_counter() - t0
    meta = {
        "camera_id": camera_id,
        "source": str(video),
        "num_frames": done,
        "fps": src.fps,
        "detector": detector.cfg.weights.name,
        "device": detector.device,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "index_fps": round(done / elapsed, 2),
        "detect_fps": round(done / infer_s, 2) if infer_s else None,
    }
    writer.save(out, meta)
    if hook is not None:
        reid_meta = {**meta, "reid_backend": reid_cfg.backend, "reid_arch": reid_cfg.arch,
                     "samples": len(reid_writer.rows)}  # fmt: skip
        reid_writer.save(reid_cache_path(out.parent.parent, camera_id), reid_meta)
        meta["reid_samples"] = len(reid_writer.rows)
    if face_hook is not None:
        face_writer.save(
            face_cache_path(out.parent.parent, camera_id), {**meta, "face_model": face_cfg.model}
        )
        meta["face_samples"] = len(face_writer.rows)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cameras", nargs="*", help="default: all cached cameras with a file source")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--batch", type=int, help="frames per forward pass")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-reid", action="store_true", help="skip Re-ID embeddings")
    parser.add_argument("--no-faces", action="store_true", help="skip face samples")
    parser.add_argument("--cache-dir", type=Path, help="write here instead of paths.cache_dir (smoke tests)")
    parser.add_argument(
        "--reid-backend", choices=["osnet", "colorhist"], help="override reid.backend"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    config = get_config()
    s = config.settings
    det_cfg = s.pipeline.detector
    if args.weights:
        det_cfg = det_cfg.model_copy(update={"weights": args.weights.resolve()})
    cams = [
        c for c in config.cameras.cameras
        if c.source_type == "file" and (c.id in args.cameras if args.cameras else c.run_mode == "cached")
    ]  # fmt: skip
    if not cams:
        log.error("no matching file cameras in cameras.yaml")
        return 1
    device = resolve_device(s.device)
    detector = YoloDetector(det_cfg, device, half=s.half_precision)
    batch = args.batch or (det_cfg.batch_size * 2 if device == "cuda" else 1)
    encoder = None
    reid_cfg = (
        s.reid
        if not args.reid_backend
        else s.reid.model_copy(update={"backend": args.reid_backend})
    )
    if not args.no_reid:
        encoder = build_encoder(reid_cfg, device, s.half_precision)
    face_encoder = None if args.no_faces else build_face_encoder(s.face, device)
    for cam in cams:
        video = Path(cam.source_uri)
        video = video if video.is_absolute() else config.root_dir / video
        tracker = ByteTracker(s.pipeline.tracker, s.pipeline.min_box_height_px)
        out = cache_path(args.cache_dir or s.paths.cache_dir, cam.id)
        meta = index_camera(
            cam.id, video, detector, out, batch, args.max_frames, tracker,
            encoder, reid_cfg, face_encoder, s.face,
        )  # fmt: skip
        log.info("%s: wrote %s  (%d frames, %.1f fps end-to-end, detector %.1f fps on %s)",
                 cam.id, out, meta["num_frames"], meta["index_fps"], meta["detect_fps"] or 0, device)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
