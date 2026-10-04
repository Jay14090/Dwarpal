"""MEVA KF1 parsing (KPF YAML annotations: geom.yml + types.yml).

Lines are YAML flow mappings, one per detection. We parse with regexes instead of a YAML
loader: geom files reach hundreds of thousands of lines and safe_load is ~50x slower.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# 2018-03-11.13-50-01.13-55-01.school.G328.r13.avi
CLIP_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})\.(?P<start>\d{2}-\d{2}-\d{2})\.(?P<end>\d{2}-\d{2}-\d{2})"
    r"\.(?P<site>[a-z]+)\.(?P<cam>G\d+)"
)
GEOM_RE = re.compile(
    r"id1:\s*(?P<tid>\d+).*?ts0:\s*(?P<frame>\d+).*?g0:\s*"
    r"(?P<x1>-?[\d.]+)\s+(?P<y1>-?[\d.]+)\s+(?P<x2>-?[\d.]+)\s+(?P<y2>-?[\d.]+)"
)
TYPES_RE = re.compile(r"id1:\s*(?P<tid>\d+)\s*,\s*cset3:\s*\{\s*(?P<cls>\w+)\s*:")

CLASS_MAP = {"Person": "person", "Vehicle": "vehicle"}


@dataclass(frozen=True)
class MevaClip:
    date: str
    start: str
    end: str
    site: str
    camera: str  # e.g. "G328"

    @property
    def stem(self) -> str:
        return f"{self.date}.{self.start}.{self.end}.{self.site}.{self.camera}"

    @property
    def camera_number(self) -> int:
        return int(self.camera[1:])


def parse_clip_name(name: str) -> MevaClip:
    m = CLIP_RE.search(name)
    if not m:
        raise ValueError(f"not a MEVA clip name: {name}")
    return MevaClip(m["date"], m["start"], m["end"], m["site"], m["cam"])


def parse_types(path: Path) -> dict[int, str]:
    """track id -> our class name ("person" / "vehicle"); other KPF classes are dropped."""
    out: dict[int, str] = {}
    for line in path.read_text().splitlines():
        m = TYPES_RE.search(line)
        if m and m["cls"] in CLASS_MAP:
            out[int(m["tid"])] = CLASS_MAP[m["cls"]]
    return out


def parse_geom(path: Path) -> list[tuple[int, int, float, float, float, float]]:
    """Rows (frame, track_id, x1, y1, x2, y2) in source pixels."""
    rows: list[tuple[int, int, float, float, float, float]] = []
    for line in path.read_text().splitlines():
        if "geom:" not in line:
            continue
        m = GEOM_RE.search(line)
        if m is None:
            raise ValueError(f"unparseable geom line in {path.name}: {line[:120]}")
        rows.append(
            (
                int(m["frame"]),
                int(m["tid"]),
                float(m["x1"]),
                float(m["y1"]),
                float(m["x2"]),
                float(m["y2"]),
            )
        )
    return rows


def global_id(clip: MevaClip, track_id: int) -> int:
    """MEVA ids are per clip; namespace them by camera so they never collide (G328, 7 -> 328007)."""
    return clip.camera_number * 1000 + track_id
