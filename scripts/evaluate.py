"""Compute metrics from real runs against ground truth (`make eval`).

P3: single-camera and multi-camera IDF1 of the global-ID pipeline, replaying cached tracks and
Re-ID samples (built by `make index`) through the live GlobalTracker code.
P4: role-label accuracy and unknown-alert precision/recall on frames after the simulated
enrollment window (needs scripts/simulate_enrollment.py first).
Results are printed and saved to data/eval/<dataset>_{mtmc,identity}.json (metrics page).

Usage:
  python scripts/evaluate.py [--dataset smartspaces] [--start-frame N] [--end-frame N]
                             [--no-calibration] [--sweep match_threshold=0.4,0.5,0.6]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from app.core.config import get_config
from app.datasets.common import read_manifest
from app.eval.identity_eval import evaluate_identity
from app.eval.metrics import IdMetrics
from app.eval.mtmc import evaluate_mtmc, load_cameras
from app.pipeline.identity import Gallery
from app.pipeline.person_hooks import SampleCache, face_cache_path


def fmt(m: IdMetrics) -> str:
    return (
        f"IDF1 {100 * m.idf1:5.1f}  IDP {100 * m.idp:5.1f}  IDR {100 * m.idr:5.1f}  "
        f"ids gt/pred {m.gt_ids}/{m.pred_ids}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", default="smartspaces")
    parser.add_argument("--cameras", nargs="*", help="default: every camera in the manifest")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--no-calibration", action="store_true", help="appearance-only ablation")
    parser.add_argument("--sweep", help="global_tracker field=v1,v2,... to compare settings")
    parser.add_argument("--skip-identity", action="store_true", help="only the P3 IDF1 metrics")
    args = parser.parse_args()

    config = get_config()
    s = config.settings
    ddir = s.paths.processed_dir / args.dataset
    if not (ddir / "manifest.json").is_file():
        print(f"{args.dataset}: not found in {s.paths.processed_dir}. "
              f"Run `make data-{args.dataset}` then `make index`.", file=sys.stderr)  # fmt: skip
        return 2
    manifest = read_manifest(ddir)
    if not (manifest["exhaustive"] and manifest["cross_camera_ids"]):
        print(f"{args.dataset}: GT is not exhaustive with cross-camera ids "
              f"({manifest.get('note', '')}); refusing to report tracking metrics.", file=sys.stderr)  # fmt: skip
        return 2
    cam_ids = args.cameras or [c["id"] for c in manifest["cameras"]]
    try:
        cams = load_cameras(ddir, s.paths.cache_dir, cam_ids)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    variants: list[tuple[str, object]] = [("config", s.global_tracker)]
    if args.sweep:
        key, values = args.sweep.split("=", 1)
        variants = [
            (
                f"{key}={v}",
                s.global_tracker.model_copy(update={key: type(getattr(s.global_tracker, key))(v)}),
            )
            for v in values.split(",")
        ]
    report: dict = {
        "dataset": args.dataset,
        "cameras": cam_ids,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "calibration": not args.no_calibration,
        "track_cache": {c.camera_id: c.tracks.meta for c in cams},
        "runs": [],
    }
    for name, gcfg in variants:
        t0 = time.perf_counter()
        res = evaluate_mtmc(
            cams, s.reid, gcfg, args.start_frame, args.end_frame,
            use_calibration=not args.no_calibration, track_buffer=s.pipeline.tracker.track_buffer,
        )  # fmt: skip
        print(
            f"\n[{name}] frames {res.frames[0]}-{res.frames[1]}  ({time.perf_counter() - t0:.1f}s)"
        )
        for cam, m in res.per_camera.items():
            print(f"  {cam:<12} single-cam  {fmt(m)}")
        print(f"  {'ALL':<12} single-cam  {fmt(res.single_camera)}")
        print(f"  {'ALL':<12} multi-cam   {fmt(res.multi_camera_online)}   (online, as displayed)")
        print(f"  {'ALL':<12} multi-cam   {fmt(res.multi_camera_tracklet)}   (tracklet-level)")
        print(f"  global identities created: {res.global_ids}")
        report["runs"].append({"name": name, "global_tracker": gcfg.model_dump(), **res.as_dict()})
    out = Path(s.paths.data_dir) / "eval" / f"{args.dataset}_mtmc.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nsaved {out}")
    if not args.skip_identity:
        evaluate_roles(args, ddir, cams)
    return 0


def evaluate_roles(args: argparse.Namespace, ddir: Path, cams: list) -> None:
    s = get_config().settings
    enroll_dir = ddir / "enrollment"
    if not (enroll_dir / "enrollment.json").is_file():
        print("\nidentity metrics: no simulated enrollment yet "
              "(run `uv run python scripts/simulate_enrollment.py --residents 15 --staff 5 --seed 42`)")  # fmt: skip
        return
    info = json.loads((enroll_dir / "enrollment.json").read_text())
    gallery = Gallery.load_npz(enroll_dir / "gallery.npz")
    roles = {p["gt_id"]: p["role"] for p in info["people"]}
    faces = {
        c.camera_id: SampleCache(face_cache_path(s.paths.cache_dir, c.camera_id))
        for c in cams
        if face_cache_path(s.paths.cache_dir, c.camera_id).is_file()
    }
    start = max(info["window_end_frame"], args.start_frame)
    res = evaluate_identity(
        cams, s.reid, s.global_tracker, s.identity, s.face, gallery, roles, start,
        face_caches=faces, end_frame=args.end_frame, track_buffer=s.pipeline.tracker.track_buffer,
    )  # fmt: skip
    print(f"\n[identity] held-out frames {res.frames[0]}-{res.frames[1]} "
          f"({len(roles)} enrolled: {sum(r == 'resident' for r in roles.values())} residents, "
          f"{sum(r == 'staff' for r in roles.values())} staff; face caches: {len(faces)})")  # fmt: skip
    print(f"  role-label accuracy  {100 * res.role_accuracy:5.1f}%  over {res.decided_obs} decided observations "
          f"(coverage {100 * res.coverage:.1f}% of {res.matched_obs} matched)")  # fmt: skip
    print(f"  unknown alerts       precision {100 * res.unknown_precision:5.1f}% ({res.alerts_correct}/{res.alerts})  "
          f"recall {100 * res.unknown_recall:5.1f}% ({res.unknown_people_alerted}/{res.unknown_people} unknown people)")  # fmt: skip
    if res.false_unknown_people:
        print(f"  enrolled people wrongly alerted as unknown (GT ids): {res.false_unknown_people}")
    print("  confusion (true -> predicted):")
    for true_role, row in res.confusion.items():
        print(f"    {true_role:<9} {row}")
    out = Path(s.paths.data_dir) / "eval" / f"{args.dataset}_identity.json"
    out.write_text(json.dumps({"dataset": args.dataset, "enrollment": info | {"people": len(info["people"])},
                               "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **res.as_dict()}, indent=1))  # fmt: skip
    print(f"saved {out}")


if __name__ == "__main__":
    sys.exit(main())
