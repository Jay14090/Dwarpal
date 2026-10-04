"""Convert the downloaded MEVA subset to the common format (data/processed/meva/).

MEVA GT is activity-centric (only actors in annotated activities have boxes), so the output is
flagged exhaustive=false / cross_camera_ids=false and the evaluator will not score against it.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.core.config import get_config
from app.datasets import meva
from app.datasets.common import (
    GroundTruth,
    output_size,
    probe_video,
    transcode,
    write_manifest,
)

log = logging.getLogger("adapt_meva")


def find_one(folder: Path, pattern: str) -> Path | None:
    hits = sorted(folder.glob(pattern))
    return hits[0] if hits else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-encode existing outputs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = get_config().settings
    mcfg = cfg.datasets.meva
    raw = cfg.paths.raw_dir / "meva"
    out = cfg.paths.processed_dir / "meva"
    cameras = []

    for cam_key in mcfg.cameras:  # e.g. "school.G328"
        video = find_one(raw / "videos", f"{mcfg.clip_prefix}*.{cam_key}.r13.avi")
        if video is None:
            log.error("missing video for %s; run scripts/download_meva.sh", cam_key)
            return 1
        clip = meva.parse_clip_name(video.name)
        slug = f"meva_{clip.camera.lower()}"
        info = probe_video(video)
        w, h = output_size(info.width, info.height, cfg.datasets.max_height)
        sx, sy = w / info.width, h / info.height
        max_frames = round(mcfg.max_seconds * info.fps) if mcfg.max_seconds else info.num_frames

        dst = out / f"{slug}.mp4"
        if args.force or not dst.exists():
            log.info("transcoding %s -> %s (%dx%d)", video.name, dst.name, w, h)
            transcode(
                video, dst, w, h, info.fps, crf=cfg.datasets.crf, max_seconds=mcfg.max_seconds
            )
        out_info = probe_video(dst)

        rows: list[list[object]] = []
        geom = find_one(raw / "annotations", f"{clip.stem}.geom.yml")
        types_path = find_one(raw / "annotations", f"{clip.stem}.types.yml")
        if geom and types_path:
            types = meva.parse_types(types_path)
            for frame, tid, x1, y1, x2, y2 in meva.parse_geom(geom):
                if frame >= max_frames or tid not in types:
                    continue
                gid = meva.global_id(clip, tid)
                box = [round(x1 * sx, 1), round(y1 * sy, 1), round(x2 * sx, 1), round(y2 * sy, 1)]
                rows.append([frame, gid, *box, types[tid]])
            rows.sort(key=lambda r: (r[0], r[1]))
        else:
            log.warning("no annotations for %s; writing empty GT", clip.stem)

        gt = GroundTruth(
            dataset="meva",
            camera=slug,
            source=video.name,
            fps=out_info.fps,
            width=out_info.width,
            height=out_info.height,
            num_frames=out_info.num_frames,
            exhaustive=False,
            cross_camera_ids=False,
            rows=rows,
        )
        gt.save(out / "gt" / f"{slug}.json")
        cameras.append(
            {
                "id": slug,
                "source_video": video.name,
                "site": clip.site,
                "start_time": f"{clip.date}T{clip.start.replace('-', ':')}",
                "calibration": None,
            }
        )
        log.info(
            "%s: %d frames, %d GT boxes, persons=%d vehicles=%d",
            slug, gt.num_frames, len(rows), len(gt.identities("person")), len(gt.identities("vehicle")),
        )  # fmt: skip

    write_manifest(
        out,
        {
            "dataset": "meva",
            "source": f"{mcfg.bucket}/{mcfg.video_prefix}",
            "license": "MEVA terms (https://mevadata.org)",
            "exhaustive": False,
            "cross_camera_ids": False,
            "note": "GT covers only actors in annotated activities; not usable for tracking metrics.",
            "cameras": cameras,
        },
    )
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
