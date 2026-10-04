"""Download one SmartSpaces scene subset: ground truth, calibration and the N busiest cameras.

Steps (CLAUDE.md download rules):
  1. List the scene's files with sizes (no download).
  2. Download ground_truth.txt + calibration (small), rank cameras by person count and
     cross-camera overlap, pick `num_cameras`.
  3. Print the selection and its size; above `download_confirm_gb` require --yes.
  4. Download video.mp4 + calibration.json for the selected cameras.

Needs network access to huggingface.co. If the dataset is gated for your account, accept its
terms on the dataset page and set HF_TOKEN in .env.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

from app.core.config import get_config
from app.datasets.smartspaces import CAMERA_DIR_RE, identity_sets, pick_cameras

GB = 1e9
WANTED_FILE = re.compile(r"/(video\.mp4|calibration\.json)$")


def human(n: float) -> str:
    return f"{n / 1e6:9.1f} MB" if n < GB else f"{n / GB:9.2f} GB"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--scene", help="override settings.datasets.smartspaces.scene")
    parser.add_argument("--num-cameras", type=int, help="override num_cameras")
    parser.add_argument("--list-only", action="store_true", help="only list files and sizes")
    parser.add_argument("--yes", action="store_true", help="confirm downloads above the threshold")
    args = parser.parse_args()

    cfg = get_config().settings
    scfg = cfg.datasets.smartspaces
    scene = (args.scene or scfg.scene).strip("/")
    n_cams = args.num_cameras or scfg.num_cameras
    out_root = cfg.paths.raw_dir / "smartspaces"
    token = os.environ.get("HF_TOKEN") or None
    api = HfApi()

    files = [
        f
        for f in api.list_repo_tree(
            scfg.repo_id, path_in_repo=scene, recursive=True, repo_type="dataset", token=token
        )
        if isinstance(f, RepoFile)
    ]
    if not files:
        print(f"no files under {scene}", file=sys.stderr)
        return 1
    by_path = {f.path: f.size for f in files}
    print(f"{scfg.repo_id}/{scene}: {len(files)} files, {human(sum(by_path.values()))} total")
    for path, size in sorted(by_path.items()):
        print(f"  {human(size)}  {path[len(scene) + 1 :]}")
    if args.list_only:
        return 0

    def fetch(path: str) -> Path:
        return Path(
            hf_hub_download(
                scfg.repo_id, path, repo_type="dataset", local_dir=out_root, token=token
            )
        )

    gt_path = fetch(f"{scene}/ground_truth.txt")
    for extra in ("calibration_2025_format.json",):
        if f"{scene}/{extra}" in by_path:
            fetch(f"{scene}/{extra}")

    available = {
        int(m.group(1))
        for p in by_path
        for part in p.split("/")
        if (m := CAMERA_DIR_RE.match(part))
    }
    ids = {c: s for c, s in identity_sets(gt_path).items() if c in available}
    chosen = pick_cameras(ids, n_cams, exclude=set(scfg.exclude_cameras))
    union = set().union(*(ids[c] for c in chosen)) if chosen else set()
    print(f"\nselected {len(chosen)} cameras ({len(union)} distinct people across them):")
    wanted: list[str] = []
    for c in chosen:
        cam_dir = f"{scene}/camera_{c:04d}"
        cam_files = [p for p in by_path if p.startswith(cam_dir + "/") and WANTED_FILE.search(p)]
        wanted += cam_files
        size = sum(by_path[p] for p in cam_files)
        print(f"  camera_{c:04d}: {len(ids[c]):3d} people  {human(size)}")
    total = sum(by_path[p] for p in wanted)
    print(f"download size: {human(total)}")
    if total / GB > cfg.datasets.download_confirm_gb and not args.yes:
        print(f"over {cfg.datasets.download_confirm_gb} GB: re-run with --yes to confirm")
        return 1

    for p in wanted:
        print(f"  fetching {p}")
        fetch(p)
    sel = out_root / scene / "selection.json"
    sel.write_text(json.dumps({"scene": scene, "cameras": chosen}, indent=2))
    print(f"done -> {out_root / scene}  (selection saved to {sel.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
