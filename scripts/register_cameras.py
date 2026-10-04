"""Append processed dataset cameras that are missing from config/cameras.yaml.

New entries are file sources in cached mode with one full-frame, non-restricted zone; edit the
zones afterwards. The file is appended to as text so existing comments survive.

Usage: python scripts/register_cameras.py smartspaces [--enable N]
"""

from __future__ import annotations

import argparse
import sys

from app.core.config import get_config, load_config
from app.datasets.common import read_manifest

TEMPLATE = """
  - id: {id}
    name: {name}
    source_type: file
    source_uri: {uri}
    run_mode: cached
    enabled: {enabled}
    dataset: {dataset}
    zones:
      - name: {id}_floor
        restricted: false
        polygon: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    calibration: null
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--enable", type=int, default=2, help="enable the first N new cameras")
    args = parser.parse_args()

    config = get_config()
    ddir = config.settings.paths.processed_dir / args.dataset
    manifest = read_manifest(ddir)
    existing = {c.id for c in config.cameras.cameras}
    path = config.config_dir / "cameras.yaml"
    added = 0
    with path.open("a", encoding="utf-8") as fh:
        for cam in manifest["cameras"]:
            if cam["id"] in existing:
                continue
            uri = (ddir / f"{cam['id']}.mp4").relative_to(config.root_dir)
            fh.write(
                TEMPLATE.format(
                    id=cam["id"],
                    name=f"{args.dataset} {cam['id']}",
                    uri=uri.as_posix(),
                    enabled="true" if added < args.enable else "false",
                    dataset=args.dataset,
                )
            )
            added += 1
    load_config(config.config_dir)  # fail loudly if the result does not validate
    print(f"added {added} cameras to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
