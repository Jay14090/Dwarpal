"""Common internal dataset format shared by adapters, the pipeline and the evaluator.

Layout under `data/processed/<dataset>/`:
    <camera>.mp4                 H.264 video (<= max_height, 1 s GOP, no B-frames)
    gt/<camera>.json             ground truth (see GroundTruth)
    calibration/<camera>.json    camera model (see Calibration), absent if unknown
    manifest.json                dataset-level info (cameras, source, flags)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

GT_COLUMNS = ["frame", "global_id", "x1", "y1", "x2", "y2", "class"]


@dataclass
class GroundTruth:
    """Per-camera boxes. `rows` are [frame, global_id, x1, y1, x2, y2, class] in output pixels.

    exhaustive: every visible object of the listed classes is annotated (safe to score FP/IDs).
    cross_camera_ids: the same global_id means the same object in every camera of the dataset.
    """

    dataset: str
    camera: str
    fps: float
    width: int
    height: int
    num_frames: int
    exhaustive: bool
    cross_camera_ids: bool
    rows: list[list[Any]] = field(default_factory=list)
    source: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "camera": self.camera,
            "source": self.source,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "exhaustive": self.exhaustive,
            "cross_camera_ids": self.cross_camera_ids,
            "columns": GT_COLUMNS,
            "rows": self.rows,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> GroundTruth:
        if data.get("columns") != GT_COLUMNS:
            raise ValueError(f"unexpected GT columns {data.get('columns')}")
        return cls(
            dataset=data["dataset"],
            camera=data["camera"],
            source=data.get("source", ""),
            fps=float(data["fps"]),
            width=int(data["width"]),
            height=int(data["height"]),
            num_frames=int(data["num_frames"]),
            exhaustive=bool(data["exhaustive"]),
            cross_camera_ids=bool(data["cross_camera_ids"]),
            rows=data["rows"],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), separators=(",", ":")))

    @classmethod
    def load(cls, path: Path) -> GroundTruth:
        return cls.from_json(json.loads(path.read_text()))

    # ------------------------------------------------------------------ queries

    def by_frame(self) -> dict[int, list[tuple[int, tuple[float, float, float, float], str]]]:
        out: dict[int, list[tuple[int, tuple[float, float, float, float], str]]] = defaultdict(list)
        for f, gid, x1, y1, x2, y2, cls_ in self.rows:
            out[int(f)].append((int(gid), (float(x1), float(y1), float(x2), float(y2)), cls_))
        return out

    def identities(self, cls_: str | None = None) -> set[int]:
        return {int(r[1]) for r in self.rows if cls_ is None or r[6] == cls_}

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps if self.fps else 0.0


@dataclass
class Calibration:
    """Pinhole camera model in output-pixel units. World units are meters, ground plane z = 0.

    P = K [R | t] maps homogeneous world points to the image.
    H maps ground-plane points (x, y, 1) to the image (columns 0, 1, 3 of P).
    """

    camera: str
    width: int
    height: int
    P: np.ndarray
    K: np.ndarray | None = None
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    source: str = ""

    @property
    def H(self) -> np.ndarray:
        return self.P[:, [0, 1, 3]]

    def scaled(self, sx: float, sy: float, width: int, height: int) -> Calibration:
        """Calibration for the same camera after resizing the image by (sx, sy)."""
        S = np.diag([sx, sy, 1.0])
        return Calibration(
            camera=self.camera,
            width=width,
            height=height,
            P=S @ self.P,
            K=None if self.K is None else S @ self.K,
            R=self.R,
            t=self.t,
            source=self.source,
        )

    def project(self, world: np.ndarray) -> np.ndarray:
        """Project Nx3 world points to Nx2 pixels."""
        pts = np.hstack([np.asarray(world, dtype=float), np.ones((len(world), 1))])
        img = (self.P @ pts.T).T
        return img[:, :2] / img[:, 2:3]

    def image_to_ground(self, pixels: np.ndarray) -> np.ndarray:
        """Back-project Nx2 pixels onto the ground plane (z = 0); returns Nx2 world xy."""
        pts = np.hstack([np.asarray(pixels, dtype=float), np.ones((len(pixels), 1))])
        world = (np.linalg.inv(self.H) @ pts.T).T
        return world[:, :2] / world[:, 2:3]

    def to_json(self) -> dict[str, Any]:
        def arr(a: np.ndarray | None) -> Any:
            return None if a is None else np.asarray(a).tolist()

        return {
            "camera": self.camera,
            "width": self.width,
            "height": self.height,
            "P": arr(self.P),
            "H": arr(self.H),
            "K": arr(self.K),
            "R": arr(self.R),
            "t": arr(self.t),
            "units": "m",
            "source": self.source,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Calibration:
        def arr(key: str) -> np.ndarray | None:
            v = data.get(key)
            return None if v is None else np.asarray(v, dtype=float)

        P = arr("P")
        if P is None or P.shape != (3, 4):
            raise ValueError("calibration needs a 3x4 P matrix")
        return cls(
            camera=data["camera"],
            width=int(data["width"]),
            height=int(data["height"]),
            P=P,
            K=arr("K"),
            R=arr("R"),
            t=arr("t"),
            source=data.get("source", ""),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: Path) -> Calibration:
        return cls.from_json(json.loads(path.read_text()))


def write_manifest(dataset_dir: Path, info: dict[str, Any]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "manifest.json").write_text(json.dumps(info, indent=2))


def read_manifest(dataset_dir: Path) -> dict[str, Any]:
    return json.loads((dataset_dir / "manifest.json").read_text())


# ---------------------------------------------------------------------- video helpers


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    num_frames: int
    duration_s: float
    codec: str


def probe_video(path: Path) -> VideoInfo:
    """Read stream metadata with ffprobe (counts frames from the container when available)."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    data = json.loads(out.stdout)
    st = data["streams"][0]
    num, den = (int(x) for x in st["r_frame_rate"].split("/"))
    fps = num / den if den else 0.0
    duration = float(data.get("format", {}).get("duration") or 0.0)
    nb = st.get("nb_frames")
    frames = int(nb) if nb and nb != "N/A" else round(duration * fps)
    return VideoInfo(int(st["width"]), int(st["height"]), fps, frames, duration, st["codec_name"])


def output_size(width: int, height: int, max_height: int) -> tuple[int, int]:
    """Target size keeping aspect ratio, height <= max_height, both even (H.264 requirement)."""
    if height <= max_height:
        w, h = width, height
    else:
        h = max_height
        w = round(width * max_height / height)
    return w - (w % 2), h - (h % 2)


def transcode(
    src: Path,
    dst: Path,
    width: int,
    height: int,
    fps: float,
    crf: int = 23,
    max_seconds: float | None = None,
    preset: str = "veryfast",
) -> None:
    """Re-encode to an RTSP-friendly H.264 file: fixed 1 s GOP, no B-frames, no audio."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    gop = max(1, round(fps))
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src)]
    if max_seconds:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += [
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0", "-bf", "0",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst),
    ]  # fmt: skip
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
