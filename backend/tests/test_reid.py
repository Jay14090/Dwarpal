from __future__ import annotations

import numpy as np
import pytest
import torch

from app.datasets.common import Calibration
from app.pipeline.crosscam import CrossCameraHook
from app.pipeline.frames import Frame, FrameResult, Track
from app.pipeline.global_tracker import GlobalTracker
from app.pipeline.reid import (
    OSNET_CHANNELS,
    ColorHistEncoder,
    OsnetEncoder,
    crop_quality,
    load_torchreid_checkpoint,
)
from app.vendor.osnet import OSBlock, OSNet


def osnet(arch="osnet_x0_25", classes=1000):
    return OSNet(classes, blocks=[OSBlock] * 3, layers=[2, 2, 2], channels=OSNET_CHANNELS[arch])


def save_torchreid_style(path, model):
    """torchreid saves {'state_dict': ..., 'epoch': ...}, often with DataParallel 'module.' keys."""
    torch.save(
        {"state_dict": {f"module.{k}": v for k, v in model.state_dict().items()}, "epoch": 60}, path
    )


def test_checkpoint_loader_handles_torchreid_layout(tmp_path):
    src = osnet(classes=4101)  # MSMT17 has 4101 ids: classifier shape differs from ours
    save_torchreid_style(tmp_path / "w.pt", src)
    dst = osnet()
    missing = load_torchreid_checkpoint(dst, tmp_path / "w.pt")
    assert missing == []
    k = "conv1.conv.weight"
    assert torch.equal(dst.state_dict()[k], src.state_dict()[k])


def test_checkpoint_loader_reports_wrong_architecture(tmp_path):
    save_torchreid_style(tmp_path / "w.pt", osnet("osnet_x0_25"))
    assert load_torchreid_checkpoint(osnet("osnet_x1_0"), tmp_path / "w.pt")  # shapes mismatch


def test_osnet_encoder_embeds_normalized_512d(tmp_path, config):
    save_torchreid_style(tmp_path / "w.pt", osnet())
    cfg = config.settings.reid.model_copy(
        update={"arch": "osnet_x0_25", "weights": tmp_path / "w.pt"}
    )
    enc = OsnetEncoder(cfg, "cpu", half=False)
    crops = [np.full((120, 50, 3), v, np.uint8) for v in (30, 200)] + [
        np.zeros((0, 0, 3), np.uint8)
    ]
    e = enc.embed(crops)
    assert e.shape == (3, 512)
    np.testing.assert_allclose(np.linalg.norm(e[:2], axis=1), 1.0, rtol=1e-5)


def test_missing_weights_without_url_fails_clearly(tmp_path, config):
    cfg = config.settings.reid.model_copy(
        update={"weights": tmp_path / "nope.pt", "weights_url": ""}
    )
    with pytest.raises(FileNotFoundError, match=r"nope\.pt"):
        OsnetEncoder(cfg, "cpu")


def test_crop_quality_prefers_big_clear_crops():
    big = crop_quality((100, 100, 180, 340), 0.9, (1280, 720))
    small = crop_quality((100, 100, 120, 160), 0.9, (1280, 720))
    cut = crop_quality((0, 100, 80, 340), 0.9, (1280, 720))  # touches the border
    hidden = crop_quality((100, 100, 180, 340), 0.9, (1280, 720), others=[(100, 100, 180, 300)])
    assert big > cut > 0 and small == 0.0 and hidden < 0.2 * big


def test_hook_projects_foot_point_to_ground(config):
    # camera 5 m above the origin looking straight down: pixel (u, v) <-> ground (x, y)
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    R = np.diag([1.0, -1.0, -1.0])
    t = -R @ np.array([0.0, 0.0, 5.0])
    cal = Calibration("cam", 640, 480, K @ np.hstack([R, t[:, None]]), K, R, t)
    gt = GlobalTracker(config.settings.global_tracker)
    hook = CrossCameraHook(
        "cam", config.settings.reid, gt, encoder=ColorHistEncoder(), calibration=cal
    )
    img = np.zeros((480, 640, 3), np.uint8)
    track = Track(1, (300.0, 100.0, 340.0, 340.0), 0.9, "person")  # foot point (320, 340)
    hook(FrameResult(Frame("cam", 0, 0.0, img), [track]))
    x, y = track.extra["world_xy"]
    assert x == pytest.approx(0.0, abs=1e-6) and y == pytest.approx(
        -1.0, abs=1e-6
    )  # 100 px * 5 m / 500 px
    assert gt.sightings[("cam", 1)].positions[-1][1:] == pytest.approx((0.0, -1.0))
