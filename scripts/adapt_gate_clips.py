"""Turn your own gate CCTV clips (data/raw/gate_vehicles/*) into ANPR cameras.

Transcodes each clip to data/processed/gate/gate_NN.mp4 (same RTSP-friendly H.264 as the
dataset adapters, but kept at up to 1080p: plates need pixels), writes a manifest, and appends
the cameras to config/cameras.yaml with `anpr: true`, run_mode realtime, enabled.
"""

from __future__ import annotations

import argparse
import sys

from app.core.config import get_config, load_config
from app.datasets.common import output_size, probe_video, transcode, write_manifest

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".ts", ".m4v"}
TEMPLATE = """
  - id: {id}
    name: Gate camera {n}
    source_type: file
    source_uri: {uri}
    run_mode: realtime
    enabled: true
    anpr: true
    dataset: gate
    zones:
      - name: {id}_entry
        restricted: false
        polygon: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    calibration: null
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-height", type=int, default=1080)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    config = get_config()
    s = config.settings
    src_dir = s.paths.raw_dir / "gate_vehicles"
    clips = (
        sorted(p for p in src_dir.glob("*") if p.suffix.lower() in VIDEO_EXT)
        if src_dir.is_dir()
        else []
    )
    if not clips:
        print(f"no clips in {src_dir}: copy your gate CCTV videos there first", file=sys.stderr)
        return 1
    out = s.paths.processed_dir / "gate"
    cams = []
    for i, clip in enumerate(clips, 1):
        info = probe_video(clip)
        w, h = output_size(info.width, info.height, args.max_height)
        cam_id = f"gate_{i:02d}"
        dst = out / f"{cam_id}.mp4"
        if not dst.exists():
            print(f"transcoding {clip.name} -> {dst.name} ({w}x{h})")
            transcode(clip, dst, w, h, info.fps or 25.0, crf=s.datasets.crf)
        cams.append({"id": cam_id, "source_video": clip.name, "calibration": None})
    write_manifest(out, {"dataset": "gate", "source": str(src_dir), "exhaustive": False, "cross_camera_ids": False,
                         "note": "user's own gate clips; no ground truth", "cameras": cams})  # fmt: skip
    if args.no_register:
        return 0
    existing = {c.id for c in config.cameras.cameras}
    path = config.config_dir / "cameras.yaml"
    with path.open("a", encoding="utf-8") as fh:
        for i, cam in enumerate(cams, 1):
            if cam["id"] not in existing:
                uri = (out / f"{cam['id']}.mp4").relative_to(config.root_dir).as_posix()
                fh.write(TEMPLATE.format(id=cam["id"], n=i, uri=uri))
    load_config(config.config_dir)  # validate
    print(f"{len(cams)} gate camera(s) ready; registered in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
