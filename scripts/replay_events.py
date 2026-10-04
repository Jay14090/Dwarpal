"""Replay cached cameras through identity + rules and check alerts fire once per incident (P8 DoD).

Runs the same hooks as the live engine (global IDs -> identity state machine -> rules engine) over
cached detections, without pacing. The gallery comes from Postgres (enrolled people) unless
--no-db. Alongside the rules engine, an independent tally records every (person, zone) presence
interval of resolved people in each rule's zones (gap <= presence_gap_s keeps one interval),
which is what "one alert per incident" is checked against:
  - no (rule, person) pair gets two alerts within its cooldown;
  - no presence interval gets more than one alert of the same rule.

Usage: python scripts/replay_events.py [--cameras meva_g328 ...] [--start 2026-10-04T22:30:00]
       [--reid-backend colorhist] [--max-frames N] [--no-db]
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from app.core.config import get_config
from app.core.device import resolve_device
from app.datasets.common import read_manifest
from app.events import EventBus, EventRecord
from app.pipeline.cache import TrackCache, cache_path
from app.pipeline.crosscam import CrossCameraHook, ReidCache, reid_cache_path
from app.pipeline.engine import load_calibration_for
from app.pipeline.frames import FrameResult
from app.pipeline.global_tracker import GlobalTracker
from app.pipeline.identity import Gallery, IdentityEngine
from app.pipeline.person_hooks import IdentityHook
from app.pipeline.reid import build_encoder
from app.pipeline.sources import VideoFileSource
from app.rules import DEFAULT_ROLES, RulesEngine, RulesHook, person_key
from app.zones import zones_at

log = logging.getLogger("replay_events")


def clip_start(config, cam) -> float | None:  # noqa: ANN001
    mpath = config.settings.paths.processed_dir / (cam.dataset or "") / "manifest.json"
    if not cam.dataset or not mpath.is_file():
        return None
    entry = next((c for c in read_manifest(mpath.parent)["cameras"] if c["id"] == cam.id), {})
    return (
        datetime.fromisoformat(entry["start_time"]).timestamp() if entry.get("start_time") else None
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cameras", nargs="*")
    ap.add_argument(
        "--start",
        help="local ISO time of frame 0 (default: clip start from the manifest, else now)",
    )
    ap.add_argument(
        "--reid-backend",
        choices=["osnet", "colorhist"],
        help="encoder when a camera has no reid.npz",
    )
    ap.add_argument("--max-frames", type=int)
    ap.add_argument(
        "--no-db", action="store_true", help="empty gallery: everyone resolves to unknown"
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    config = get_config()
    s = config.settings
    cams = [
        c for c in config.cameras.cameras
        if c.source_type == "file" and cache_path(s.paths.cache_dir, c.id).is_file()
        and (c.id in args.cameras if args.cameras else c.enabled and c.run_mode == "cached")
    ]  # fmt: skip
    if not cams:
        log.error("no cached file cameras")
        return 1
    gallery = Gallery()
    if not args.no_db:
        from sqlalchemy.orm import Session

        from app.db.gallery_store import load_gallery
        from app.db.session import make_engine

        with Session(make_engine(s.database)) as session:
            gallery = load_gallery(session)
    log.info("gallery: %d enrolled people", len(gallery.people))

    events: list[EventRecord] = []
    bus = EventBus(config.rules, events.append)
    gap = config.rules.defaults.presence_gap_s
    rules = RulesEngine(config.rules, bus, s.app.timezone, gap, thumb=lambda *_: None)
    gtracker = GlobalTracker(s.global_tracker, s.reid.max_samples, 1)
    identity = IdentityEngine(s.identity, gallery)
    encoder = None

    if args.start:
        base = datetime.fromisoformat(args.start).timestamp()
    else:
        base = clip_start(config, cams[0]) or time.time()
    runs = []
    for cam in cams:
        video = Path(cam.source_uri)
        src = VideoFileSource(
            cam.id,
            video if video.is_absolute() else config.root_dir / video,
            loop=False,
            paced=False,
        )
        rp = reid_cache_path(s.paths.cache_dir, cam.id)
        if rp.is_file():
            cc = CrossCameraHook(
                cam.id,
                s.reid,
                gtracker,
                cache=ReidCache(rp),
                calibration=load_calibration_for(cam, config),
            )
        else:
            if encoder is None:
                rcfg = s.reid.model_copy(update={"backend": args.reid_backend or s.reid.backend})
                encoder = build_encoder(rcfg, resolve_device(s.device), s.half_precision)
            cc = CrossCameraHook(
                cam.id,
                s.reid,
                gtracker,
                encoder=encoder,
                calibration=load_calibration_for(cam, config),
            )
        runs.append(
            (
                cam,
                src,
                TrackCache(cache_path(s.paths.cache_dir, cam.id)),
                [cc, IdentityHook(identity), RulesHook(cam, rules)],
            )
        )

    # independent incident tally: (rule, key) -> list of [start, last] presence intervals
    intervals: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    n = min(min(r[1].num_frames, r[2].num_frames) for r in runs)
    if args.max_frames:
        n = min(n, args.max_frames)
    t0 = time.perf_counter()
    for f in range(n):
        for cam, src, tracks, hooks in runs:
            frame = src.read()
            if frame is None:
                continue
            frame.ts = base + f / src.fps
            result = FrameResult(frame, tracks.tracks_at(frame.index))
            for hook in hooks:
                hook(result)
            h, w = frame.image.shape[:2]
            for t in result.tracks:
                if not t.is_person:
                    continue
                for rule in rules.rules:
                    if t.role not in rule.params.get(
                        "roles", DEFAULT_ROLES.get(rule.type, ["unknown"])
                    ):
                        continue
                    zs = set(zones_at(cam, t.xyxy, w, h)) & rules.zones_for(rule, cam)
                    if zs:
                        iv = intervals[(rule.id, person_key(cam.id, t))]
                        if iv and frame.ts - iv[-1][1] <= gap:
                            iv[-1][1] = frame.ts
                        else:
                            iv.append([frame.ts, frame.ts])
        if f and f % 1500 == 0:
            log.info("frame %d/%d", f, n)

    by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for e in events:
        key = (
            f"g{e.global_id}"
            if e.global_id is not None
            else f"{e.camera_id}:t{e.payload['track_id']}"
        )
        by_pair[(e.rule, key)].append(e.ts)
    cooldown_violations = sum(
        1 for (rule, _), ts in by_pair.items() for a, b in itertools.pairwise(ts)
        if b - a < config.rules.cooldown_for(next(r for r in config.rules.rules if r.id == rule))
    )  # fmt: skip
    multi = 0
    for (rule, key), ivs in intervals.items():
        for a, b in ivs:
            k = sum(1 for ts in by_pair.get((rule, key), []) if a <= ts <= b + 1e-6)
            multi += k > 1
    n_incidents = sum(len(v) for v in intervals.values())
    roles = defaultdict(int)
    for st in identity.states.values():
        roles[st.role_state] += 1
    report = {
        "cameras": [c.id for c in cams],
        "frames_per_camera": n,
        "seconds": round(time.perf_counter() - t0, 1),
        "global_ids": len(identity.states),
        "identity_states": dict(roles),
        "events": len(events),
        "events_by_rule": {
            r: sum(1 for e in events if e.rule == r) for r in sorted({e.rule for e in events})
        },
        "incidents_eligible": n_incidents,
        "incidents_with_more_than_one_alert": multi,
        "cooldown_violations": cooldown_violations,
        "event_list": [
            {
                "rule": e.rule,
                "camera": e.camera_id,
                "global_id": e.global_id,
                "local_time": datetime.fromtimestamp(e.ts, rules.tz).strftime("%H:%M:%S"),
                **{k: e.payload[k] for k in ("zone", "role", "dwell_s")},
            }
            for e in events
        ],
    }
    print(json.dumps(report, indent=1))
    out = s.paths.data_dir / "eval" / "replay_events.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"saved {out}")
    return 0 if multi == 0 and cooldown_violations == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
