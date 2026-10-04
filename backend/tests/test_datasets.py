from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.datasets import meva, smartspaces
from app.datasets.common import Calibration, GroundTruth, output_size, read_manifest

FIXTURES = Path(__file__).parent / "fixtures"
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

# Real lines from 2018-03-11.13-50-01.13-55-01.school.G328.{geom,types}.yml
MEVA_GEOM = """\
- { meta: "2018-03-11.13-50-01.13-55-01.school.G328 geometry"}
- { meta: "7 tracks; 1056 detections"}
- { geom: { id1: 1, id0: 0, ts0: 2760, ts1: 92, g0: 867 733 1076 816 , src: truth, } }
- { geom: { id1: 1, id0: 1, ts0: 2761, ts1: 92.0333333333333, g0: 862 733 1071 817 , src: truth, } }
- { geom: { id1: 5, id0: 2, ts0: 2762, ts1: 92.0666666666667, g0: 857 733 1066 818 , src: truth, } }
"""
MEVA_TYPES = """\
- { meta: "2018-03-11.13-50-01.13-55-01.school.G328 types"}
- { types: { id1: 1 , cset3: { Vehicle: 1.0 } } }
- { types: { id1: 5 , cset3: { Person: 1.0 } } }
- { types: { id1: 9 , cset3: { Other: 1.0 } } }
"""


def test_meva_parsers(tmp_path):
    (tmp_path / "g.yml").write_text(MEVA_GEOM)
    (tmp_path / "t.yml").write_text(MEVA_TYPES)
    rows = meva.parse_geom(tmp_path / "g.yml")
    assert rows[0] == (2760, 1, 867.0, 733.0, 1076.0, 816.0)
    assert len(rows) == 3
    assert meva.parse_types(tmp_path / "t.yml") == {1: "vehicle", 5: "person"}


def test_meva_clip_name_and_ids():
    clip = meva.parse_clip_name("2018-03-11.13-50-01.13-55-01.school.G328.r13.avi")
    assert (clip.date, clip.site, clip.camera) == ("2018-03-11", "school", "G328")
    assert clip.stem == "2018-03-11.13-50-01.13-55-01.school.G328"
    assert meva.global_id(clip, 7) == 328007
    with pytest.raises(ValueError):
        meva.parse_clip_name("random.avi")


def test_meva_bad_geom_line_raises(tmp_path):
    (tmp_path / "g.yml").write_text("- { geom: { id1: 1, broken } }\n")
    with pytest.raises(ValueError, match="unparseable"):
        meva.parse_geom(tmp_path / "g.yml")


def test_output_size_keeps_aspect_and_even():
    assert output_size(1920, 1080, 720) == (1280, 720)
    assert output_size(1920, 1072, 720) == (1290, 720)
    assert output_size(640, 480, 720) == (640, 480)
    assert output_size(641, 481, 720) == (640, 480)


def real_calibration() -> Calibration:
    return smartspaces.load_calibration(FIXTURES / "smartspaces" / "scene_071", 635, 1920, 1080)


def test_calibration_projection_roundtrip():
    cal = real_calibration()
    assert cal.K is not None and cal.R is not None
    ground = np.array([[0.0, 0.0], [-10.0, 5.0], [2.5, -3.0]])
    pix = cal.project(np.hstack([ground, np.zeros((3, 1))]))
    np.testing.assert_allclose(cal.image_to_ground(pix), ground, atol=1e-6)


def test_calibration_scaling_matches_resized_image():
    cal = real_calibration()
    small = cal.scaled(1280 / 1920, 720 / 1080, 1280, 720)
    world = np.array([[-10.0, 5.0, 1.7]])
    np.testing.assert_allclose(small.project(world), cal.project(world) * [1280 / 1920, 720 / 1080])
    loaded = Calibration.from_json(small.to_json())
    np.testing.assert_allclose(loaded.P, small.P)


def test_calibration_falls_back_to_per_camera_file(tmp_path):
    scene = tmp_path / "scene"
    shutil.copytree(FIXTURES / "smartspaces" / "scene_071" / "camera_0635", scene / "camera_0635")
    cal = smartspaces.load_calibration(scene, 635, 1920, 1080)
    assert cal.K is None
    np.testing.assert_allclose(cal.P, real_calibration().P)


def test_pick_cameras_prefers_overlap():
    ids = {1: {1, 2, 3, 4, 5}, 2: {1, 2, 3}, 3: {10, 11, 12, 13}, 4: {4, 5, 6}}
    assert smartspaces.pick_cameras(ids, 2) == [1, 2]
    assert smartspaces.pick_cameras(ids, 3) == [1, 2, 4]
    assert smartspaces.pick_cameras(ids, 2, exclude={1}) == [3, 2]


def make_video(path: Path, seconds: int = 2, size: str = "1920x1080") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=30",
         "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )  # fmt: skip


@needs_ffmpeg
def test_adapt_smartspaces_scene(tmp_path):
    import importlib.util

    scene = tmp_path / "raw" / "MTMC_Tracking_2024" / "test" / "scene_071"
    shutil.copytree(FIXTURES / "smartspaces" / "scene_071", scene)
    shutil.copytree(scene / "camera_0635", scene / "camera_0636")
    make_video(scene / "camera_0635" / "video.mp4")
    make_video(scene / "camera_0636" / "video.mp4")
    (scene / "ground_truth.txt").write_text(
        "635 7 0 100 200 50 150 1.0 2.0\n"
        "636 7 0 300 300 60 120 1.0 2.0\n"
        "635 8 59 960 540 30 90 0.0 0.0\n"
        "635 9 500 0 0 10 10 0 0\n"  # beyond the 2 s video: dropped
    )
    spec = importlib.util.spec_from_file_location(
        "adapt_smartspaces", Path(__file__).parents[2] / "scripts" / "adapt_smartspaces.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = tmp_path / "processed"
    mod.adapt(scene, out, max_height=720, crf=30, max_seconds=None)

    manifest = read_manifest(out)
    assert manifest["exhaustive"] and manifest["cross_camera_ids"]
    assert [c["id"] for c in manifest["cameras"]] == ["ss_0635", "ss_0636"]
    gt = GroundTruth.load(out / "gt" / "ss_0635.json")
    assert (gt.width, gt.height, gt.num_frames) == (1280, 720, 60)
    s = 720 / 1080
    assert gt.rows[0] == [
        0,
        7,
        round(100 * s, 1),
        round(200 * s, 1),
        round(150 * s, 1),
        round(350 * s, 1),
        "person",
    ]
    assert gt.identities() == {7, 8}
    assert GroundTruth.load(out / "gt" / "ss_0636.json").identities() == {7}
    cal = Calibration.load(out / "calibration" / "ss_0635.json")
    assert (cal.width, cal.height) == (1280, 720)
    np.testing.assert_allclose(cal.P, np.diag([s, s, 1.0]) @ real_calibration().P)
