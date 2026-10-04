"""Convert a downloaded SmartSpaces scene to the common format (data/processed/smartspaces/).

Reads data/raw/smartspaces/<scene>/ (ground_truth.txt, calibration, camera_XXXX/video.mp4),
transcodes each camera to <= max_height, rescales GT boxes and calibration accordingly.
GT is exhaustive with cross-camera identities, so it is used for all tracking/identity metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from app.core.config import get_config
from app.datasets.common import GroundTruth, output_size, probe_video, transcode, write_manifest
from app.datasets.smartspaces import CAMERA_DIR_RE, camera_slug, iter_ground_truth, load_calibration

log = logging.getLogger("adapt_smartspaces")


def adapt(scene_dir: Path, out: Path, max_height: int, crf: int, max_seconds: float | None,
          force: bool = False, cameras: list[int] | None = None) -> dict:  # fmt: skip
    if cameras is None:
        sel = scene_dir / "selection.json"
        if sel.is_file():
            cameras = json.loads(sel.read_text())["cameras"]
        else:
            cameras = sorted(
                int(m.group(1))
                for d in scene_dir.iterdir()
                if (m := CAMERA_DIR_RE.match(d.name)) and (d / "video.mp4").is_file()
            )
    if not cameras:
        raise SystemExit(f"no cameras with video.mp4 under {scene_dir}")

    # Group GT rows per camera in one pass over the (large) scene file.
    raw_rows: dict[int, list[tuple[int, int, float, float, float, float]]] = defaultdict(list)
    wanted = set(cameras)
    for cam, obj, frame, x1, y1, x2, y2 in iter_ground_truth(scene_dir / "ground_truth.txt"):
        if cam in wanted:
            raw_rows[cam].append((frame, obj, x1, y1, x2, y2))

    entries = []
    for cam in cameras:
        slug = camera_slug(cam)
        src = scene_dir / f"camera_{cam:04d}" / "video.mp4"
        info = probe_video(src)
        w, h = output_size(info.width, info.height, max_height)
        sx, sy = w / info.width, h / info.height
        dst = out / f"{slug}.mp4"
        if force or not dst.exists():
            log.info("transcoding %s -> %s (%dx%d)", src, dst.name, w, h)
            transcode(src, dst, w, h, info.fps, crf=crf, max_seconds=max_seconds)
        out_info = probe_video(dst)

        rows = [
            [
                f,
                gid,
                round(x1 * sx, 1),
                round(y1 * sy, 1),
                round(x2 * sx, 1),
                round(y2 * sy, 1),
                "person",
            ]
            for f, gid, x1, y1, x2, y2 in sorted(raw_rows.get(cam, []))
            if f < out_info.num_frames
        ]
        gt = GroundTruth(
            dataset="smartspaces", camera=slug, source=str(src.relative_to(scene_dir.parent)),
            fps=out_info.fps, width=out_info.width, height=out_info.height,
            num_frames=out_info.num_frames, exhaustive=True, cross_camera_ids=True, rows=rows,
        )  # fmt: skip
        gt.save(out / "gt" / f"{slug}.json")

        calib = load_calibration(scene_dir, cam, info.width, info.height)
        calib.scaled(sx, sy, out_info.width, out_info.height).save(
            out / "calibration" / f"{slug}.json"
        )
        entries.append(
            {"id": slug, "source_camera": cam, "calibration": f"calibration/{slug}.json"}
        )
        log.info(
            "%s: %d frames, %d boxes, %d people",
            slug,
            gt.num_frames,
            len(rows),
            len(gt.identities()),
        )

    manifest = {
        "dataset": "smartspaces",
        "source": f"nvidia/PhysicalAI-SmartSpaces/{scene_dir.relative_to(scene_dir.parents[2])}",
        "license": "CC-BY-4.0",
        "exhaustive": True,
        "cross_camera_ids": True,
        "cameras": entries,
    }
    write_manifest(out, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, help="default: raw_dir/smartspaces/<scene>")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = get_config().settings
    scfg = cfg.datasets.smartspaces
    scene_dir = args.scene_dir or cfg.paths.raw_dir / "smartspaces" / scfg.scene
    if not (scene_dir / "ground_truth.txt").is_file():
        log.error("missing %s/ground_truth.txt; run scripts/download_smartspaces.py", scene_dir)
        return 1
    adapt(scene_dir, cfg.paths.processed_dir / "smartspaces", cfg.datasets.max_height,
          cfg.datasets.crf, scfg.max_seconds, force=args.force)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
