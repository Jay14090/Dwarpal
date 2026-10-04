"""Index cached dataset cameras into Postgres for search (`make index-db`, P6).

Replays every cached file camera frame by frame (synchronized across cameras, real decoded
frames) through the live hooks: global IDs (reid.npz), faces (faces.npz), identity roles (gallery
from Postgres), then TrackIndexer -> tracks, track_embeddings (CLIP + body), thumbnails and
global_identities. Timestamps are base_time + frame / fps, where base_time is the clip's real
start time when the manifest has one (MEVA), else --base-time.

Usage: python scripts/index_tracks.py [--cameras ...] [--base-time 2026-10-04T18:00:00] [--replace]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import get_config
from app.core.device import resolve_device
from app.datasets.common import read_manifest
from app.db.gallery_store import load_gallery
from app.db.models import Track
from app.db.session import make_engine
from app.db.sync import sync_cameras
from app.db.track_store import TrackStore, next_global_id
from app.pipeline.cache import TrackCache, cache_path
from app.pipeline.clip import ClipEncoder
from app.pipeline.crosscam import CrossCameraHook, ReidCache, reid_cache_path
from app.pipeline.engine import load_calibration_for
from app.pipeline.frames import FrameResult
from app.pipeline.global_tracker import GlobalTracker
from app.pipeline.identity import IdentityEngine
from app.pipeline.indexer import DbWriter, TrackIndexer
from app.pipeline.person_hooks import FaceHook, IdentityHook, SampleCache, face_cache_path
from app.pipeline.sources import VideoFileSource

log = logging.getLogger("index_tracks")


def camera_start(config, cam) -> float | None:  # noqa: ANN001
    if not cam.dataset:
        return None
    mpath = config.settings.paths.processed_dir / cam.dataset / "manifest.json"
    if not mpath.is_file():
        return None
    entry = next((c for c in read_manifest(mpath.parent)["cameras"] if c["id"] == cam.id), {})
    st = entry.get("start_time")
    return datetime.fromisoformat(st).timestamp() if st else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--cameras", nargs="*", help="default: every cached file camera with a cache"
    )
    parser.add_argument(
        "--base-time", help="ISO datetime for frame 0 when the dataset has no real time"
    )
    parser.add_argument(
        "--replace", action="store_true", help="delete existing tracks of these cameras first"
    )
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    config = get_config()
    s = config.settings
    cams = [
        c for c in config.cameras.cameras
        if c.source_type == "file" and cache_path(s.paths.cache_dir, c.id).is_file()
        and (c.id in args.cameras if args.cameras else c.run_mode == "cached" and c.enabled)
    ]  # fmt: skip
    if not cams:
        log.error("no cached file cameras with a track cache (run `make index` first)")
        return 1
    db = make_engine(s.database)
    with Session(db) as session:
        sync_cameras(session, config.cameras)
        if args.replace:
            n = session.execute(
                delete(Track).where(Track.camera_id.in_([c.id for c in cams]))
            ).rowcount
            log.info("removed %d existing tracks", n)
        session.commit()
        start_id = next_global_id(session)
        gallery = load_gallery(session)
    writer = DbWriter(db, max_queue=100000)
    store = TrackStore(writer, s.paths.thumbs_dir)
    gtracker = GlobalTracker(s.global_tracker, s.reid.max_samples, start_id)
    gtracker.new_identity_listeners.append(
        lambda ident: store.identity_created(ident.id, ident.first_seen)
    )
    identity = IdentityEngine(s.identity, gallery)
    identity.listeners.append(
        lambda st, prev, ts: store.identity_changed(st.global_id, st.role_state, st.person_id, ts)
    )
    device = resolve_device(s.device)
    clip = ClipEncoder(s.clip, device, s.half_precision)

    default_base = (
        datetime.fromisoformat(args.base_time).timestamp() if args.base_time else time.time()
    )
    runs = []
    for cam in cams:
        video = Path(cam.source_uri)
        src = VideoFileSource(
            cam.id,
            video if video.is_absolute() else config.root_dir / video,
            loop=False,
            paced=False,
        )
        tracks = TrackCache(cache_path(s.paths.cache_dir, cam.id))
        hooks = []
        rp = reid_cache_path(s.paths.cache_dir, cam.id)
        if rp.is_file():
            hooks.append(CrossCameraHook(cam.id, s.reid, gtracker, cache=ReidCache(rp),
                                         calibration=load_calibration_for(cam, config)))  # fmt: skip
        else:
            log.warning("%s: no reid cache; tracks are indexed without global ids", cam.id)
        fp = face_cache_path(s.paths.cache_dir, cam.id)
        if fp.is_file():
            hooks.append(FaceHook(cam.id, s.face, cache=SampleCache(fp)))
        hooks.append(IdentityHook(identity))
        indexer = TrackIndexer(cam, s.index, store.store, clip=clip, clip_top_k=s.clip.top_k,
                               calibration=load_calibration_for(cam, config),
                               grace_frames=int(s.pipeline.tracker.track_buffer + src.fps))  # fmt: skip
        hooks.append(indexer)
        base = camera_start(config, cam) or default_base
        runs.append((cam, src, tracks, hooks, indexer, base))

    n_frames = min(min(r[1].num_frames, r[2].num_frames) for r in runs)
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)
    t0 = time.perf_counter()
    for f in range(n_frames):
        for _cam, src, tracks, hooks, _, base in runs:
            frame = src.read()
            if frame is None:
                continue
            frame.ts = base + f / src.fps
            result = FrameResult(frame, tracks.tracks_at(frame.index))
            for hook in hooks:
                hook(result)
        if f and f % 1500 == 0:
            log.info(
                "frame %d/%d (%.1f frames/s per camera)",
                f,
                n_frames,
                f / (time.perf_counter() - t0),
            )
    for *_, indexer, _ in runs:
        indexer.flush()
    writer.flush(timeout=120)
    writer.close()
    with Session(db) as session:
        total = session.scalar(
            select(Track.id)
            .where(Track.camera_id.in_([c.id for c in cams]))
            .order_by(Track.id.desc())
            .limit(1)
        )
        n = len(
            list(session.scalars(select(Track.id).where(Track.camera_id.in_([c.id for c in cams]))))
        )
    log.info("indexed %d frames x %d cameras in %.0fs; %d tracks in DB for these cameras (last id %s); db failures %d",
             n_frames, len(runs), time.perf_counter() - t0, n, total, writer.failures)  # fmt: skip
    return 0 if writer.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
