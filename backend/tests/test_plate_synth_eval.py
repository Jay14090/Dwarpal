from __future__ import annotations

import random

from app.anpr.normalize import normalize
from app.anpr.synth import random_plate, render_plate, synthetic_scene


def test_random_plates_are_valid_indian_formats():
    rng = random.Random(0)
    plates = [random_plate(rng) for _ in range(500)]
    assert all(normalize(p).valid and normalize(p).text == p for p in plates)
    assert any("BH" in p[2:4] for p in plates)


def test_render_and_scene_shapes():
    rng = random.Random(1)
    img = render_plate("TN09AB1234", rng, height=60)
    assert img.shape[0] == 60 and img.shape[1] > 3 * 60 and img.dtype.name == "uint8"
    text, scene, (x1, y1, x2, y2) = synthetic_scene(rng)
    assert (
        normalize(text).valid
        and scene.shape == (480, 640, 3)
        and 0 <= x1 < x2 <= 640
        and 0 <= y1 < y2 <= 480
    )


def test_text_metrics_definitions():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "eval_plates", Path(__file__).parents[2] / "scripts" / "eval_plates.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.text_metrics(
        [("TN09AB1234", "TN O9 AB 1234"), ("MH12AB1234", "MH12AB1235"), ("DL1CAB1234", "")]
    )
    assert m["n"] == 3
    assert abs(m["exact_plate_accuracy"] - 1 / 3) < 1e-9  # O->0 fixed by normalization
    assert m["exact_plate_accuracy_raw_ocr"] == 0.0
    d = mod.det_metrics(
        [[(0, 0, 10, 10)], [(0, 0, 10, 10)]], [[(0, 0, 10, 10), (50, 50, 60, 60)], []]
    )
    assert (d["tp"], d["fp"], d["fn"]) == (1, 1, 1)
