"""Simulated enrollment for a dataset with exhaustive GT (CLAUDE.md section 4).

Picks ground-truth people seen in the early *enrollment window*, assigns roles (residents,
staff; everyone else is unknown), extracts their best crops from that window across cameras,
embeds them (OSNet body + faces when visible) and stores them as enrolled people.
Evaluation (`make eval`) only scores frames after the window, so there is no leakage.

Writes:
  Postgres people/person_embeddings (unit "SIM:<dataset>", replaced on every run) unless --no-db
  data/processed/<dataset>/enrollment/{gallery.npz, enrollment.json}

Usage: python scripts/simulate_enrollment.py --residents 15 --staff 5 --seed 42 [--dataset smartspaces]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_config
from app.core.device import resolve_device
from app.datasets.common import GroundTruth, read_manifest
from app.pipeline.face import build_face_encoder
from app.pipeline.identity import Gallery, GalleryPerson
from app.pipeline.reid import build_encoder, crop

log = logging.getLogger("simulate_enrollment")
MIN_HEIGHT_PX = 100  # crops shorter than this are poor enrollment material
MIN_BOXES = 30  # a candidate must be clearly visible for at least ~1 s in the window
MIN_GAP_FRAMES = 15  # spread the chosen shots in time


def pick_shots(
    boxes: list[tuple[str, int, tuple[float, ...]]], n: int
) -> list[tuple[str, int, tuple[float, ...]]]:
    """Tallest boxes first, at least MIN_GAP_FRAMES apart within a camera."""
    chosen: list[tuple[str, int, tuple[float, ...]]] = []
    for cam, f, box in sorted(boxes, key=lambda b: b[2][3] - b[2][1], reverse=True):
        if all(c != cam or abs(f - g) >= MIN_GAP_FRAMES for c, g, _ in chosen):
            chosen.append((cam, f, box))
        if len(chosen) == n:
            break
    return chosen


def read_frames(video: Path, frames: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    out = {}
    for f in sorted(set(frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            out[f] = img
    cap.release()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", default="smartspaces")
    parser.add_argument("--residents", type=int, default=15)
    parser.add_argument("--staff", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--window-frac", type=float, help="override identity.enrollment.window_frac"
    )
    parser.add_argument("--no-db", action="store_true", help="only write gallery.npz (ids 1..n)")
    parser.add_argument("--reid-backend", choices=["osnet", "colorhist"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = get_config()
    s = config.settings
    ecfg = s.identity.enrollment
    ddir = s.paths.processed_dir / args.dataset
    manifest = read_manifest(ddir)
    if not manifest["exhaustive"]:
        log.error(
            "%s GT is not exhaustive; simulated enrollment needs complete identities", args.dataset
        )
        return 2
    cams = [c["id"] for c in manifest["cameras"]]
    gts = {c: GroundTruth.load(ddir / "gt" / f"{c}.json") for c in cams}
    n_frames = min(g.num_frames for g in gts.values())
    window_end = int((args.window_frac or ecfg.window_frac) * n_frames)

    boxes: dict[int, list[tuple[str, int, tuple[float, ...]]]] = defaultdict(list)
    seen_after: set[int] = set()
    for cam, gt in gts.items():
        for f, gid, x1, y1, x2, y2, cls_ in gt.rows:
            if cls_ != "person":
                continue
            if f < window_end and y2 - y1 >= MIN_HEIGHT_PX:
                boxes[gid].append((cam, f, (x1, y1, x2, y2)))
            elif f >= window_end:
                seen_after.add(gid)
    candidates = sorted(g for g, b in boxes.items() if len(b) >= MIN_BOXES)
    want = args.residents + args.staff
    if len(candidates) < want:
        log.warning(
            "only %d people are clearly visible in the window; enrolling all of them",
            len(candidates),
        )
    rng = random.Random(args.seed)
    chosen = rng.sample(candidates, min(want, len(candidates)))
    n_res = min(args.residents, len(chosen))
    roles = {g: ("resident" if i < n_res else "staff") for i, g in enumerate(chosen)}
    log.info("window: frames 0-%d of %d; %d candidates; %d residents + %d staff",
             window_end, n_frames, len(candidates), n_res, len(chosen) - n_res)  # fmt: skip

    device = resolve_device(s.device)
    reid_cfg = (
        s.reid.model_copy(update={"backend": args.reid_backend}) if args.reid_backend else s.reid
    )
    body_enc = build_encoder(reid_cfg, device, s.half_precision)
    try:
        face_enc = build_face_encoder(s.face, device)
    except Exception as exc:  # faces are optional for datasets
        log.warning("faces unavailable (%s); body only", exc)
        face_enc = None

    shots = {g: pick_shots(boxes[g], ecfg.shots_per_person) for g in chosen}
    need: dict[str, list[int]] = defaultdict(list)
    for g in chosen:
        for cam, f, _ in shots[g]:
            need[cam].append(f)
    frames = {cam: read_frames(ddir / f"{cam}.mp4", fs) for cam, fs in need.items()}

    embs: dict[int, dict[str, list[np.ndarray]]] = {}
    for g in chosen:
        body, face = [], []
        for cam, f, box in shots[g]:
            img = frames[cam].get(f)
            if img is None:
                continue
            body.append(body_enc.embed([crop(img, box)])[0])
            if face_enc is not None:
                fs = face_enc.faces_for(img, [box])[0]
                if fs is not None:
                    face.append(fs.emb)
        embs[g] = {"body": body, "face": face}

    people: list[dict] = []
    if args.no_db:
        ids = {g: i + 1 for i, g in enumerate(chosen)}
    else:
        from sqlalchemy.orm import Session

        from app.db.gallery_store import delete_people_where_unit, enroll_person
        from app.db.session import make_engine

        unit = f"SIM:{args.dataset}"
        ids = {}
        with Session(make_engine(s.database)) as session:
            removed = delete_people_where_unit(session, unit)
            for i, g in enumerate(chosen):
                p = enroll_person(
                    session, role=roles[g], display_name=f"Sim {roles[g]} {i + 1:02d} (gt {g})", unit=unit,
                    consent=True, face=embs[g]["face"], body=embs[g]["body"], actor="simulate_enrollment",
                )  # fmt: skip
                ids[g] = p.id
            session.commit()
        log.info(
            "database: removed %d previous simulated people, enrolled %d", removed, len(chosen)
        )

    gallery = Gallery(
        {ids[g]: GalleryPerson(ids[g], roles[g], f"gt {g}") for g in chosen},
        {
            "body": [(ids[g], e) for g in chosen for e in embs[g]["body"]],
            "face": [(ids[g], e) for g in chosen for e in embs[g]["face"]],
        },
    )
    out = ddir / "enrollment"
    gallery.save_npz(out / "gallery.npz")
    for g in chosen:
        people.append({"person_id": ids[g], "gt_id": g, "role": roles[g],
                       "body_shots": len(embs[g]["body"]), "face_shots": len(embs[g]["face"])})  # fmt: skip
    info = {
        "dataset": args.dataset,
        "seed": args.seed,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "window_end_frame": window_end,
        "num_frames": n_frames,
        "reid_backend": reid_cfg.backend,
        "people": people,
        "unknown_gt_ids": sorted(seen_after - set(chosen)),
    }
    (out / "enrollment.json").write_text(json.dumps(info, indent=1))
    log.info("wrote %s (%d enrolled, %d unknown people appear after the window; faces found for %d)",
             out, len(chosen), len(info["unknown_gt_ids"]), sum(1 for p in people if p["face_shots"]))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
