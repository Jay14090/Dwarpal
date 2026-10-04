"""Plate detector + OCR evaluation (P5).

Datasets (folder with images/ and one of):
  labels.csv           filename,plate[,x1,y1,x2,y2]  -> exact-plate + character accuracy
                       (with boxes: OCR on the labeled crop, plus detector recall separately)
  labels/*.txt (YOLO)  boxes only (e.g. Kaggle kedarsai)  -> detection precision/recall @ IoU 0.5
  --synthetic N        generated Indian plate crops: OCR sanity check only, never the reported metric

Exact-plate accuracy = normalized prediction == normalized label (CLAUDE.md DoD metric).
Indian-LPR (arXiv 2111.06054) is not public, so the DoD number needs a text-labeled Indian set.

Usage:
  python scripts/eval_plates.py --dataset-dir data/raw/plates_eval/<name> [--limit N]
  python scripts/eval_plates.py --synthetic 500
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from app.anpr.normalize import clean, normalize
from app.anpr.ocr import build_plate_ocr
from app.anpr.plate_detector import build_plate_detector
from app.anpr.registry import levenshtein
from app.anpr.synth import synthetic_plate
from app.core.config import get_config
from app.core.device import resolve_device
from app.eval.metrics import iou_matrix

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def text_metrics(pairs: list[tuple[str, str]]) -> dict:
    """pairs: (label, raw prediction). Exact with and without Indian-format normalization."""
    n = len(pairs)
    exact_norm = sum(normalize(p).text == normalize(g).text for g, p in pairs)
    exact_raw = sum(clean(p) == clean(g) for g, p in pairs)
    chars = sum(len(clean(g)) for g, _ in pairs)
    errors = sum(levenshtein(normalize(p).text, normalize(g).text) for g, p in pairs)
    valid = sum(normalize(p).valid for _, p in pairs)
    return {
        "n": n,
        "exact_plate_accuracy": exact_norm / n if n else 0.0,
        "exact_plate_accuracy_raw_ocr": exact_raw / n if n else 0.0,
        "char_accuracy": 1 - errors / chars if chars else 0.0,
        "valid_format_rate": valid / n if n else 0.0,
    }


def det_metrics(gt: list[list[tuple]], pred: list[list[tuple]], thr: float = 0.5) -> dict:
    tp = fp = fn = 0
    for g, p in zip(gt, pred, strict=True):
        if not p:
            fn += len(g)
            continue
        if not g:
            fp += len(p)
            continue
        iou = iou_matrix(np.array(p, float), np.array(g, float))
        matched_g: set[int] = set()
        for i in np.argsort(-iou.max(axis=1)):
            j = int(np.argmax(iou[i]))
            if iou[i, j] >= thr and j not in matched_g:
                matched_g.add(j)
                tp += 1
            else:
                fp += 1
        fn += len(g) - len(matched_g)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0}  # fmt: skip


def read_yolo(label: Path, w: int, h: int) -> list[tuple]:
    boxes = []
    for line in label.read_text().split("\n"):
        parts = line.split()
        if len(parts) >= 5:
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            boxes.append(
                ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            )
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-dir", type=Path)
    src.add_argument("--synthetic", type=int, metavar="N")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    s = get_config().settings
    device = resolve_device(s.device)
    detector = build_plate_detector(s.anpr.detector, device)
    ocr = build_plate_ocr(s.anpr.ocr, device)
    report: dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "detector": s.anpr.detector.model_dump(mode="json"),
                    "ocr": s.anpr.ocr.model_dump(mode="json")}  # fmt: skip
    t0 = time.perf_counter()

    if args.synthetic:
        rng = random.Random(args.seed)
        crops = [synthetic_plate(rng) for _ in range(args.synthetic)]
        reads = ocr.read([img for _, img in crops])
        report["name"] = f"synthetic-{args.synthetic}"
        report["synthetic"] = True
        report["ocr_on_crops"] = text_metrics(
            [(t, r.text) for (t, _), r in zip(crops, reads, strict=True)]
        )
        report["note"] = (
            "synthetic plate crops: an OCR sanity check only. Plate *detection* is not evaluated on "
            "synthetic scenes (flat shapes are not representative of real vehicles)."
        )
    else:
        ddir = args.dataset_dir
        images = sorted(p for p in (ddir / "images").iterdir() if p.suffix.lower() in IMG_EXT)
        report["name"] = ddir.name
        report["synthetic"] = False
        labels_csv = ddir / "labels.csv"
        if labels_csv.is_file():
            rows = list(csv.DictReader(labels_csv.open()))[: args.limit]
            pairs_crop, pairs_e2e, gt_boxes, pred_boxes = [], [], [], []
            for row in rows:
                img = cv2.imread(str(ddir / "images" / row["filename"]))
                if img is None:
                    continue
                boxes = detector.detect(img)
                if row.get("x1"):
                    box = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
                    x1, y1, x2, y2 = (int(v) for v in box)
                    pairs_crop.append((row["plate"], ocr.read([img[y1:y2, x1:x2]])[0].text))
                    gt_boxes.append([box])
                    pred_boxes.append([b.xyxy for b in boxes])
                if boxes:
                    x1, y1, x2, y2 = (int(v) for v in boxes[0].xyxy)
                    crop = img[y1:y2, x1:x2]
                    pairs_e2e.append((row["plate"], ocr.read([crop])[0].text if crop.size else ""))
                elif row.get("x1"):
                    pairs_e2e.append((row["plate"], ""))
                else:  # images are already plate crops
                    pairs_e2e.append((row["plate"], ocr.read([img])[0].text))
            report["end_to_end"] = text_metrics(pairs_e2e)
            if pairs_crop:
                report["ocr_on_crops"] = text_metrics(pairs_crop)
                report["detection"] = det_metrics(gt_boxes, pred_boxes)
        elif (ddir / "labels").is_dir():
            gt_boxes, pred_boxes = [], []
            for img_path in images[: args.limit]:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                lab = ddir / "labels" / f"{img_path.stem}.txt"
                gt_boxes.append(read_yolo(lab, w, h) if lab.is_file() else [])
                pred_boxes.append([b.xyxy for b in detector.detect(img)])
            report["detection"] = det_metrics(gt_boxes, pred_boxes)
            report["note"] = (
                "boxes only: no plate text in this dataset, OCR accuracy not measurable"
            )
        else:
            print(f"{ddir}: needs labels.csv or labels/ (YOLO)", file=sys.stderr)
            return 2
    report["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("detector", "ocr")}, indent=1))
    out = s.paths.data_dir / "eval" / f"plates_{report['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
