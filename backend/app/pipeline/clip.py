"""CLIP image and text embeddings (OpenCLIP) for natural-language search.

Loads `clip.pretrained` (Hugging Face tag) and falls back to `clip.fallback_url`, an OpenCLIP
release checkpoint on GitHub, when the hub is unreachable. Both are ViT-B/32 with 512-d outputs.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np

from app.core.config import ClipSettings
from app.pipeline.reid import l2n

log = logging.getLogger(__name__)


class ClipEncoder:
    dim = 512

    def __init__(self, cfg: ClipSettings, device: str, half: bool = True) -> None:
        import open_clip
        import torch

        self.torch = torch
        self.device = device
        self.half = half and device == "cuda"
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                cfg.model, pretrained=cfg.pretrained, cache_dir=str(cfg.cache_dir)
            )
            source = cfg.pretrained
        except Exception as exc:
            if not cfg.fallback_url:
                raise
            path = cfg.cache_dir / Path(cfg.fallback_url).name
            if not path.is_file():
                log.warning(
                    "CLIP %s unavailable (%s); downloading fallback %s",
                    cfg.pretrained,
                    type(exc).__name__,
                    path.name,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(cfg.fallback_url, path)
            model, _, preprocess = open_clip.create_model_and_transforms(
                cfg.model, pretrained=str(path)
            )
            source = path.name
        self.model = model.eval().to(device)
        if self.half:
            self.model.half()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(cfg.model)
        log.info("CLIP %s (%s) on %s", cfg.model, source, device)

    def embed_images(self, crops: list[np.ndarray]) -> np.ndarray:
        """BGR crops -> (N, 512) L2-normalized."""
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        from PIL import Image

        x = self.torch.stack([self.preprocess(Image.fromarray(c[:, :, ::-1])) for c in crops]).to(
            self.device
        )
        with self.torch.inference_mode():
            feats = self.model.encode_image(x.half() if self.half else x)
        return l2n(feats.float().cpu().numpy())

    def embed_text(self, texts: list[str]) -> np.ndarray:
        with self.torch.inference_mode():
            feats = self.model.encode_text(self.tokenizer(texts).to(self.device))
        return l2n(feats.float().cpu().numpy())
