"""Synthetic Indian number plates (sanity checks and tests only, never reported as real metrics).

Renders random valid registrations (standard and BH series) HSRP-style: white or yellow plate,
black characters, blue "IND" strip, then random blur, noise, brightness and perspective.
"""

from __future__ import annotations

import random
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.anpr.normalize import BH_RE, STANDARD_RE, STATE_CODES

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def random_plate(rng: random.Random) -> str:
    if rng.random() < 0.1:
        tail = "".join(rng.choices(string.ascii_uppercase, k=rng.choice([1, 2])))
        p = f"{rng.randint(21, 26)}BH{rng.randint(0, 9999):04d}{tail}"
        assert BH_RE.match(p)
        return p
    state = rng.choice(sorted(STATE_CODES))
    district = str(rng.randint(1, 99)) if rng.random() < 0.8 else str(rng.randint(1, 9))
    if len(district) == 1 and rng.random() < 0.5:
        district = district.zfill(2)
    series = "".join(
        rng.choices(
            [c for c in string.ascii_uppercase if c not in "IO"], k=rng.choice([0, 1, 2, 2, 2, 3])
        )
    )
    p = f"{state}{district}{series}{rng.randint(1, 9999):04d}"
    assert STANDARD_RE.match(p)
    return p


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_plate(text: str, rng: random.Random, height: int = 80) -> np.ndarray:
    """Clean HSRP-style plate as a BGR image."""
    font = _font(int(height * 0.62))
    spaced = text if rng.random() < 0.3 else _spaced(text)
    bbox = font.getbbox(spaced)
    strip = int(height * 0.35)
    w = bbox[2] - bbox[0] + strip + int(height * 0.5)
    bg = (255, 255, 255) if rng.random() < 0.75 else (255, 205, 0)  # private / commercial
    img = Image.new("RGB", (w, height), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w - 1, height - 1], outline=(0, 0, 0), width=max(2, height // 25))
    d.rectangle([3, 3, strip, height - 4], fill=(30, 60, 170))
    small = _font(int(height * 0.18))
    d.text((5, height - int(height * 0.3)), "IND", font=small, fill=(255, 255, 255))
    d.text(
        (strip + int(height * 0.2) - bbox[0], (height - (bbox[3] - bbox[1])) / 2 - bbox[1]),
        spaced,
        font=font,
        fill=(0, 0, 0),
    )
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _spaced(text: str) -> str:
    if "BH" in text[2:4]:
        return f"{text[:2]} BH {text[4:8]} {text[8:]}"
    i = 2
    while i < len(text) and text[i].isdigit():
        i += 1
    j = i
    while j < len(text) and text[j].isalpha():
        j += 1
    return " ".join(p for p in (text[:2], text[2:i], text[i:j], text[j:]) if p)


def degrade(img: np.ndarray, rng: random.Random, strength: float = 1.0) -> np.ndarray:
    h, w = img.shape[:2]
    # perspective tilt
    d = strength * 0.08
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = [(rng.uniform(-d, d) * w, rng.uniform(-d, d) * h) for _ in range(4)]
    dst = np.float32([[x + jx, y + jy] for (x, y), (jx, jy) in zip(src, jitter, strict=True)])
    out = cv2.warpPerspective(
        img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE
    )
    # downscale (distance) then back up, blur, brightness, noise
    scale = rng.uniform(0.35, 1.0) if strength else 1.0
    small = cv2.resize(
        out, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA
    )
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    k = rng.choice([1, 3, 3, 5]) if strength else 1
    if k > 1:
        out = cv2.GaussianBlur(out, (k, k), 0)
    out = np.clip(out.astype(np.float32) * rng.uniform(0.6, 1.2) + rng.uniform(-25, 25), 0, 255)
    out += np.random.default_rng(rng.randint(0, 2**31)).normal(0, 6 * strength, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def synthetic_plate(rng: random.Random, height: int = 80) -> tuple[str, np.ndarray]:
    text = random_plate(rng)
    return text, degrade(render_plate(text, rng, height), rng)


def synthetic_scene(
    rng: random.Random, size: tuple[int, int] = (640, 480)
) -> tuple[str, np.ndarray, tuple[int, int, int, int]]:
    """A plate on a flat 'vehicle' rectangle over a noisy background; returns text, image, plate box."""
    W, H = size
    img = np.random.default_rng(rng.randint(0, 2**31)).integers(40, 120, (H, W, 3), dtype=np.uint8)
    cx1, cy1 = rng.randint(40, W // 3), rng.randint(60, H // 3)
    cx2, cy2 = rng.randint(2 * W // 3, W - 40), rng.randint(2 * H // 3, H - 30)
    color = [rng.randint(20, 230) for _ in range(3)]
    cv2.rectangle(img, (cx1, cy1), (cx2, cy2), color, -1)
    text, plate = synthetic_plate(rng, height=rng.randint(28, 60))
    ph, pw = plate.shape[:2]
    pw, ph = min(pw, cx2 - cx1 - 10), ph
    plate = cv2.resize(plate, (pw, ph))
    px = (cx1 + cx2 - pw) // 2
    py = cy2 - ph - rng.randint(5, 25)
    img[py : py + ph, px : px + pw] = plate
    return text, img, (px, py, px + pw, py + ph)
