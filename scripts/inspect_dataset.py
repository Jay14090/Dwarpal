"""Summarize processed datasets: cameras, resolution, duration, boxes and identity counts.

Usage: python scripts/inspect_dataset.py [dataset ...]   (default: every dataset in processed_dir)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from app.core.config import get_config
from app.datasets.common import GroundTruth, read_manifest


def inspect(dataset_dir: Path) -> None:
    manifest = read_manifest(dataset_dir)
    print(f"\n=== {manifest['dataset']}  ({manifest.get('source', '')})")
    print(
        f"    GT exhaustive={manifest['exhaustive']}  cross_camera_ids={manifest['cross_camera_ids']}"
        + (f"\n    note: {manifest['note']}" if manifest.get("note") else "")
    )
    header = f"    {'camera':<12} {'res':>10} {'fps':>5} {'dur(s)':>7} {'frames':>7} {'boxes':>7} {'persons':>7} {'vehicles':>8} calib"
    print(header)
    seen_in: Counter[int] = Counter()
    all_ids: set[int] = set()
    total_dur = 0.0
    for cam in manifest["cameras"]:
        gt = GroundTruth.load(dataset_dir / "gt" / f"{cam['id']}.json")
        persons, vehicles = gt.identities("person"), gt.identities("vehicle")
        for gid in gt.identities():
            seen_in[gid] += 1
        all_ids |= gt.identities()
        total_dur += gt.duration_s
        calib = "yes" if (dataset_dir / "calibration" / f"{cam['id']}.json").is_file() else "no"
        print(
            f"    {cam['id']:<12} {f'{gt.width}x{gt.height}':>10} {gt.fps:>5.1f} {gt.duration_s:>7.1f} "
            f"{gt.num_frames:>7} {len(gt.rows):>7} {len(persons):>7} {len(vehicles):>8} {calib}"
        )
    print(
        f"    cameras={len(manifest['cameras'])}  total video={total_dur / 60:.1f} min  identities={len(all_ids)}"
    )
    if manifest["cross_camera_ids"]:
        multi = sum(1 for c in seen_in.values() if c >= 2)
        print(f"    identities seen by >= 2 cameras: {multi}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*")
    args = parser.parse_args()
    root = get_config().settings.paths.processed_dir
    names = args.datasets or sorted(p.parent.name for p in root.glob("*/manifest.json"))
    if not names:
        print(f"no processed datasets in {root}; run the adapters first", file=sys.stderr)
        return 1
    for name in names:
        inspect(root / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
