from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from huggingface_hub.hf_api import RepoFile

from app.core import config as config_mod

SCRIPT = Path(__file__).parents[2] / "scripts" / "download_smartspaces.py"
SCENE = "MTMC_Tracking_2024/test/scene_071"


def load_script():
    spec = importlib.util.spec_from_file_location("download_smartspaces", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def repo_file(path: str, size: int) -> RepoFile:
    return RepoFile(path=path, size=size, oid="x")


@pytest.fixture
def fake_hub(tmp_path, monkeypatch, config_dir):
    raw = tmp_path / "data"
    (config_dir / "settings.yaml").write_text(
        (config_dir / "settings.yaml").read_text().replace("raw_dir: data/raw", f"raw_dir: {raw}")
    )
    monkeypatch.setenv("DWARPAL_CONFIG_DIR", str(config_dir))
    config_mod.get_config.cache_clear()
    gt = "".join(
        f"{cam} {pid} 0 0 0 10 10 0 0\n"
        for cam, pids in {635: [1, 2, 3], 636: [1, 2], 637: [9], 649: [1, 2, 3, 4]}.items()
        for pid in pids
    )
    files = [repo_file(f"{SCENE}/ground_truth.txt", len(gt))]
    for cam, size in {635: 2e8, 636: 1.5e8, 637: 1e8, 649: 3e8}.items():
        files.append(repo_file(f"{SCENE}/camera_{cam:04d}/video.mp4", int(size)))
        files.append(repo_file(f"{SCENE}/camera_{cam:04d}/calibration.json", 700))
    fetched: list[str] = []

    def fake_download(repo_id, filename, repo_type, local_dir, token):
        fetched.append(filename)
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(gt if filename.endswith("ground_truth.txt") else "x")
        return str(dst)

    mod = load_script()
    monkeypatch.setattr(mod.HfApi, "list_repo_tree", lambda self, *a, **k: iter(files))
    monkeypatch.setattr(mod, "hf_hub_download", fake_download)
    yield mod, fetched, raw
    config_mod.get_config.cache_clear()


def run(mod, monkeypatch, *argv) -> int:
    monkeypatch.setattr(sys, "argv", ["download_smartspaces.py", *argv])
    return mod.main()


def test_selects_busiest_overlapping_cameras_excluding_corrupt(fake_hub, monkeypatch, capsys):
    mod, fetched, raw = fake_hub
    assert run(mod, monkeypatch, "--num-cameras", "2") == 0
    videos = sorted(f for f in fetched if f.endswith("video.mp4"))
    # 649 has the most people but is excluded (corrupt per README); 635 + 636 overlap.
    assert videos == [f"{SCENE}/camera_0635/video.mp4", f"{SCENE}/camera_0636/video.mp4"]
    assert (raw / "smartspaces" / SCENE / "selection.json").is_file()
    assert "selected 2 cameras" in capsys.readouterr().out


def test_list_only_downloads_nothing(fake_hub, monkeypatch):
    mod, fetched, _ = fake_hub
    assert run(mod, monkeypatch, "--list-only") == 0
    assert fetched == []


def test_large_download_needs_confirmation(fake_hub, monkeypatch):
    mod, fetched, _ = fake_hub
    monkeypatch.setattr(mod, "GB", 1e6)  # make the 20 "GB" threshold tiny
    assert run(mod, monkeypatch) == 1
    assert not any(f.endswith("video.mp4") for f in fetched)
    assert run(mod, monkeypatch, "--yes") == 0
    assert any(f.endswith("video.mp4") for f in fetched)
