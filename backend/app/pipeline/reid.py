"""Body Re-ID: person crop -> L2-normalized 512-d embedding, plus crop quality scoring.

OsnetEncoder loads official torchreid OSNet checkpoints (architecture vendored in
app/vendor/osnet.py). ColorHistEncoder is a weights-free fallback used by tests and when
OSNet weights are unavailable; it is much weaker and is never used silently (logged loudly).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from app.core.config import ReidSettings

log = logging.getLogger(__name__)

OSNET_CHANNELS = {
    "osnet_x1_0": [64, 256, 384, 512],
    "osnet_x0_75": [48, 192, 288, 384],
    "osnet_x0_5": [32, 128, 192, 256],
    "osnet_x0_25": [16, 64, 96, 128],
}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class ReidEncoder(Protocol):
    dim: int

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """BGR crops -> (N, dim) float32, rows L2-normalized."""
        ...


def l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


def crop(
    image: np.ndarray, xyxy: tuple[float, float, float, float], pad: float = 0.0
) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = xyxy
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    return image[y1:y2, x1:x2]


def crop_quality(
    xyxy: tuple[float, float, float, float],
    conf: float,
    frame_wh: tuple[int, int],
    others: list[tuple[float, float, float, float]] | None = None,
    min_height: float = 64.0,
    full_height: float = 256.0,
) -> float:
    """Heuristic 0..1 score of how useful a person crop is for identity.

    Product of: size (height ramps from min_height to full_height), detector confidence,
    border truncation penalty, aspect ratio plausibility (standing person ~ 0.3-0.6 w/h)
    and occlusion (overlap with other boxes).
    """
    x1, y1, x2, y2 = xyxy
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    W, H = frame_wh
    size = float(np.clip((h - min_height) / max(1.0, full_height - min_height), 0.0, 1.0))
    margin = 2.0
    truncated = x1 <= margin or y1 <= margin or x2 >= W - margin or y2 >= H - margin
    border = 0.5 if truncated else 1.0
    ar = w / h
    aspect = 1.0 if 0.25 <= ar <= 0.65 else 0.5
    occl = 1.0
    for o in others or []:
        ix = max(0.0, min(x2, o[2]) - max(x1, o[0]))
        iy = max(0.0, min(y2, o[3]) - max(y1, o[1]))
        occl = min(occl, 1.0 - (ix * iy) / (w * h))
    return float(size * conf * border * aspect * max(0.0, occl))


class ColorHistEncoder:
    """HSV histogram of upper and lower body halves (dim 512). Weak; tests and fallback only."""

    dim = 512

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(crops), self.dim), np.float32)
        for i, c in enumerate(crops):
            if c.size == 0:
                continue
            hsv = cv2.cvtColor(cv2.resize(c, (64, 128)), cv2.COLOR_BGR2HSV)
            parts = (hsv[:64], hsv[64:])
            feats = [
                cv2.calcHist([p], [0, 1], None, [16, 16], [0, 180, 0, 256]).ravel() for p in parts
            ]
            out[i] = np.concatenate(feats)
        return l2n(out)


class OsnetEncoder:
    def __init__(self, cfg: ReidSettings, device: str, half: bool = True) -> None:
        import torch

        from app.vendor.osnet import OSBlock, OSNet

        if cfg.arch not in OSNET_CHANNELS:
            raise ValueError(f"unknown OSNet arch {cfg.arch}")
        self.torch = torch
        self.device = device
        self.half = half and device == "cuda"
        self.input_hw = (cfg.input_height, cfg.input_width)
        self.batch_size = cfg.batch_size
        model = OSNet(
            1000, blocks=[OSBlock] * 3, layers=[2, 2, 2], channels=OSNET_CHANNELS[cfg.arch]
        )
        self.dim = model.feature_dim
        weights = ensure_weights(cfg)
        missing = load_torchreid_checkpoint(model, weights)
        if missing:
            raise RuntimeError(f"{weights.name}: missing backbone keys {missing[:5]}...")
        model.eval().to(device)
        if self.half:
            model.half()
        self.model = model
        log.info("Re-ID %s (%s) on %s, dim=%d", cfg.arch, weights.name, device, self.dim)

    def _prep(self, crops: list[np.ndarray]) -> np.ndarray:
        h, w = self.input_hw
        batch = np.zeros((len(crops), 3, h, w), np.float32)
        for i, c in enumerate(crops):
            if c.size == 0:
                continue
            rgb = cv2.cvtColor(
                cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB
            )
            batch[i] = ((rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(
                2, 0, 1
            )
        return batch

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        torch = self.torch
        outs = []
        with torch.inference_mode():
            for i in range(0, len(crops), self.batch_size):
                x = torch.from_numpy(self._prep(crops[i : i + self.batch_size])).to(self.device)
                if self.half:
                    x = x.half()
                outs.append(self.model(x).float().cpu().numpy())
        return l2n(np.concatenate(outs))


def load_torchreid_checkpoint(model: object, path: Path) -> list[str]:
    """Load a torchreid checkpoint (raw state_dict or {'state_dict': ...}, optional 'module.'
    prefix). The classifier is skipped (its size depends on the training set). Returns the
    backbone keys that were not found (should be empty)."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k.removeprefix("module."): v for k, v in state.items()}
    own = model.state_dict()  # type: ignore[attr-defined]
    usable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(usable, strict=False)  # type: ignore[attr-defined]
    return [k for k in own if not k.startswith("classifier") and k not in usable]


def ensure_weights(cfg: ReidSettings) -> Path:
    if cfg.weights.is_file():
        return cfg.weights
    if not cfg.weights_url:
        raise FileNotFoundError(f"Re-ID weights {cfg.weights} missing and no weights_url set")
    log.info("downloading Re-ID weights %s", cfg.weights.name)
    import gdown

    cfg.weights.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(cfg.weights_url, str(cfg.weights), quiet=False)
    if not cfg.weights.is_file():
        raise FileNotFoundError(f"download of {cfg.weights_url} failed")
    return cfg.weights


def build_encoder(cfg: ReidSettings, device: str, half: bool = True) -> ReidEncoder:
    if cfg.backend == "colorhist":
        log.warning(
            "Re-ID backend is the colour-histogram fallback (weak); set reid.backend: osnet"
        )
        return ColorHistEncoder()
    return OsnetEncoder(cfg, device, half)
