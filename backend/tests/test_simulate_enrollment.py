from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from app.core import config as config_mod
from app.datasets.common import GroundTruth, write_manifest
from app.pipeline.identity import Gallery

SCRIPT = Path(__file__).parents[2] / "scripts" / "simulate_enrollment.py"
COLORS = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}


def make_dataset(ddir: Path, frames: int = 120) -> None:
    """Two cameras, three people (coloured boxes). Person 3 only appears after the window."""
    cams = []
    for cam in ("c1", "c2"):
        w = cv2.VideoWriter(
            str(ddir / f"{cam}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240)
        )
        rows = []
        for f in range(frames):
            img = np.zeros((240, 320, 3), np.uint8)
            for pid, x in ((1, 20), (2, 140), (3, 240)):
                if pid == 3 and f < 60:
                    continue
                box = (x, 40, x + 50, 200)
                cv2.rectangle(img, box[:2], box[2:], COLORS[pid], -1)
                rows.append([f, pid, *map(float, box), "person"])
            w.write(img)
        w.release()
        GroundTruth("synth", cam, 30.0, 320, 240, frames, True, True, rows).save(
            ddir / "gt" / f"{cam}.json"
        )
        cams.append({"id": cam})
    write_manifest(
        ddir, {"dataset": "synth", "exhaustive": True, "cross_camera_ids": True, "cameras": cams}
    )


def test_simulated_enrollment_end_to_end(tmp_path, config_dir, monkeypatch):
    processed = tmp_path / "processed"
    make_dataset(
        processed / "synth" if (processed / "synth").mkdir(parents=True) is None else processed
    )
    text = (config_dir / "settings.yaml").read_text()
    text = text.replace("processed_dir: data/processed", f"processed_dir: {processed}")
    text = text.replace("face:\n  enabled: true", "face:\n  enabled: false")
    (config_dir / "settings.yaml").write_text(text)
    monkeypatch.setenv("DWARPAL_CONFIG_DIR", str(config_dir))
    config_mod.get_config.cache_clear()
    spec = importlib.util.spec_from_file_location("simulate_enrollment", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    argv = ["x", "--dataset", "synth", "--residents", "1", "--staff", "1", "--seed", "0",
            "--no-db", "--reid-backend", "colorhist", "--window-frac", "0.4"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    try:
        assert mod.main() == 0
    finally:
        config_mod.get_config.cache_clear()

    info = json.loads((processed / "synth" / "enrollment" / "enrollment.json").read_text())
    assert info["window_end_frame"] == 48
    assert sorted(p["gt_id"] for p in info["people"]) == [1, 2]  # 3 is not visible in the window
    assert sorted(p["role"] for p in info["people"]) == ["resident", "staff"]
    assert info["unknown_gt_ids"] == [3]
    assert all(p["body_shots"] == 8 for p in info["people"])
    g = Gallery.load_npz(processed / "synth" / "enrollment" / "gallery.npz")
    assert len(g) == 2 and g.count("body") == 16
