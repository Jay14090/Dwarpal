"""Compute metrics from real runs against ground truth (`make eval`).

P3: single-camera and multi-camera IDF1 of the global-ID pipeline, replaying cached tracks and
Re-ID samples (built by `make index`) through the live GlobalTracker code.
Results are printed and saved to data/eval/<dataset>_mtmc.json (read by the metrics page).

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
from app.eval.metrics import IdMetrics
from app.eval.mtmc import evaluate_mtmc, load_cameras


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
