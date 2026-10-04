"""Download Indian plate data for ANPR evaluation / fine-tuning (P5).

Indian-LPR (arXiv 2111.06054, the dataset named in CLAUDE.md) is NOT public: its authors withhold
it for legal reasons (github.com/sanchit2843/Indian_LPR). Fallbacks, per CLAUDE.md:

  --kaggle kedarsai/indian-license-plates-with-labels   (CC0, ~2k images, YOLO plate *boxes*, no text)
      needs KAGGLE_USERNAME / KAGGLE_KEY (kaggle.com > Settings > API > Create New Token)
  --roboflow WORKSPACE/PROJECT/VERSION                  (any Roboflow Universe Indian plate set)
      needs ROBOFLOW_API_KEY; exported in YOLO format

Text labels (needed for exact-plate accuracy) come from your own set: put crops or full images in
data/raw/plates_eval/<name>/images/ and a labels.csv with `filename,plate[,x1,y1,x2,y2]`.
Output: data/raw/plates/<source>/ with images/ + labels/ (YOLO), usable by eval_plates.py and
notebooks/train_plate.ipynb.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from base64 import b64encode
from pathlib import Path

from app.core.config import get_config


def fetch(url: str, headers: dict[str, str] | None = None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def flatten_yolo(src: Path, dst: Path) -> int:
    """Collect every image with a same-stem .txt label into dst/images + dst/labels."""
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for img in src.rglob("*"):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lab = next(
            (
                p
                for p in (img.with_suffix(".txt"), img.parent.parent / "labels" / f"{img.stem}.txt")
                if p.is_file()
            ),
            None,
        )
        if lab is None:
            continue
        shutil.copy(img, dst / "images" / img.name)
        shutil.copy(lab, dst / "labels" / f"{img.stem}.txt")
        n += 1
    return n


def kaggle(slug: str, out: Path) -> int:
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        sys.exit("set KAGGLE_USERNAME and KAGGLE_KEY in .env (kaggle.com > Settings > API)")
    auth = b64encode(f"{user}:{key}".encode()).decode()
    print(f"downloading kaggle dataset {slug} ...")
    blob = fetch(
        f"https://www.kaggle.com/api/v1/datasets/download/{slug}",
        {"Authorization": f"Basic {auth}"},
    )
    raw = out / "_raw"
    zipfile.ZipFile(io.BytesIO(blob)).extractall(raw)
    n = flatten_yolo(raw, out)
    shutil.rmtree(raw)
    return n


def roboflow(spec: str, out: Path) -> int:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("set ROBOFLOW_API_KEY in .env")
    workspace, project, version = spec.split("/")
    meta = json.loads(
        fetch(f"https://api.roboflow.com/{workspace}/{project}/{version}/yolov8?api_key={key}")
    )
    link = meta["export"]["link"]
    raw = out / "_raw"
    zipfile.ZipFile(io.BytesIO(fetch(link))).extractall(raw)
    n = flatten_yolo(raw, out)
    shutil.rmtree(raw)
    return n


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--kaggle", metavar="OWNER/DATASET")
    g.add_argument("--roboflow", metavar="WORKSPACE/PROJECT/VERSION")
    args = parser.parse_args()
    raw_dir = get_config().settings.paths.raw_dir / "plates"
    name = (args.kaggle or args.roboflow).replace("/", "_")
    out = raw_dir / name
    n = kaggle(args.kaggle, out) if args.kaggle else roboflow(args.roboflow, out)
    print(
        f"{n} labeled images -> {out}  (evaluate: uv run python scripts/eval_plates.py --dataset-dir {out})"
    )
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
