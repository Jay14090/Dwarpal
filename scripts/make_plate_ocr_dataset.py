"""Write synthetic Indian plate crops + a fast-plate-ocr CSV (image_path,plate_text,plate_region).

Used by notebooks/train_plate.ipynb to teach the OCR the Indian layout before (or alongside)
fine-tuning on real labeled crops. Synthetic data never counts toward reported accuracy.

Usage: python scripts/make_plate_ocr_dataset.py --n 20000 --out data/processed/plates_synth
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2

from app.anpr.synth import synthetic_plate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--out", type=Path, default=Path("data/processed/plates_synth"))
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.n):
        text, img = synthetic_plate(rng)
        name = f"images/{i:06d}.jpg"
        cv2.imwrite(str(args.out / name), img)
        rows.append((name, text, "Unknown"))  # 'Unknown' keeps the pretrained region head intact
    n_val = int(len(rows) * args.val_frac)
    for split, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
        with (args.out / f"{split}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["image_path", "plate_text", "plate_region"])
            w.writerows(part)
    print(f"{len(rows)} synthetic plates -> {args.out} (train.csv / val.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
