"""Load and validate `config/settings.yaml`, `cameras.yaml` and `rules.yaml`.

Every model forbids unknown keys so a typo in YAML fails loudly at startup instead of
silently falling back to a default.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"

# Env var -> dotted path inside settings.yaml
ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "DATABASE_URL": ("database", "url"),
    "DWARPAL_DEVICE": ("device",),
    "LLM_PROVIDER": ("llm", "provider"),
    "LLM_MODEL": ("llm", "model"),
    "LLM_API_KEY": ("llm", "api_key"),
    "LLM_BASE_URL": ("llm", "base_url"),
}

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_]*$")


class ConfigError(RuntimeError):
    """Raised when a config file is missing or invalid."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- settings.yaml


class AppSettings(StrictModel):
    name: str = "dwarpal"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    timezone: str = "Asia/Kolkata"


class ServerSettings(StrictModel):
    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)
    cors_origins: list[str] = Field(default_factory=list)


class DatabaseSettings(StrictModel):
    url: str
    connect_timeout_s: int = Field(2, ge=1)
    pool_size: int = Field(5, ge=1)


class PathSettings(StrictModel):
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    cache_dir: Path
    thumbs_dir: Path
    clips_dir: Path
    models_dir: Path

    def resolved(self, root: Path) -> PathSettings:
        """Return a copy with relative paths made absolute against `root`."""
        values = {k: (v if v.is_absolute() else (root / v)) for k, v in self}
        return PathSettings(**values)


class StreamingSettings(StrictModel):
    rtsp_base_url: str = "rtsp://localhost:8554"
    mjpeg_max_fps: int = Field(15, ge=1, le=60)
    mjpeg_jpeg_quality: int = Field(80, ge=10, le=100)


class EnrollmentSettings(StrictModel):
    window_frac: float = Field(0.25, gt=0.0, lt=1.0)
    shots_per_person: int = Field(8, ge=1)
    capture_seconds: float = Field(6, gt=0)
    capture_shots: int = Field(5, ge=1)
    min_shots: int = Field(3, ge=1)


class IdentitySettings(StrictModel):
    t_face: float = Field(ge=-1.0, le=1.0)
    t_body: float = Field(ge=-1.0, le=1.0)
    min_quality: float = Field(ge=0.0, le=1.0)
    unknown_after_observations: int = Field(ge=1)
    face_weight: float = Field(ge=0.0)
    body_weight: float = Field(ge=0.0)
    accept_score: float = Field(1.0, gt=0.0)
    switch_ratio: float = Field(2.0, ge=1.0)
    recent_samples: int = Field(20, ge=1)
    staff_vest_fallback: bool = False
    enrollment: EnrollmentSettings = EnrollmentSettings()


class FaceSettings(StrictModel):
    enabled: bool = True
    model: str = "buffalo_l"
    root: Path = Path("models/insightface")
    det_size: int = Field(320, ge=64)
    det_thresh: float = Field(0.5, ge=0.0, le=1.0)
    min_person_height_px: int = Field(140, ge=0)
    min_face_px: int = Field(24, ge=1)
    good_face_px: int = Field(80, ge=1)
    sample_every: int = Field(5, ge=1)


class PrivacySettings(StrictModel):
    blur_unknown_faces: bool = True
    # identity states whose faces are blurred (anyone not identified as a consented, enrolled person)
    blur_roles: list[str] = Field(default_factory=lambda: ["unknown", "pending"])
    head_fraction: float = Field(
        0.24, gt=0.0, le=0.6
    )  # top part of the person box treated as the head
    unknown_retention_days: int = Field(7, ge=1)
    retention_interval_s: float = Field(3600.0, gt=0)
    require_consent: bool = True
    admin_actors: list[str] = Field(
        default_factory=lambda: ["admin"]
    )  # X-Actor values allowed to unblur
    unblur_max_s: float = Field(300.0, gt=0)  # a live-stream unblur expires after this


class LLMSettings(StrictModel):
    provider: str = ""
    model: str = ""
    api_key: str = Field("", repr=False)
    base_url: str = ""
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    timeout_s: int = Field(10, ge=1)

    @property
    def enabled(self) -> bool:
        return bool(self.provider)


class SmartSpacesSettings(StrictModel):
    repo_id: str = "nvidia/PhysicalAI-SmartSpaces"
    scene: str = "MTMC_Tracking_2024/test/scene_071"
    num_cameras: int = Field(6, ge=1, le=16)
    exclude_cameras: list[int] = Field(default_factory=list)
    max_seconds: float | None = Field(None, gt=0)


class MevaSettings(StrictModel):
    bucket: str = "s3://mevadata-public-01"
    video_prefix: str
    annotation_prefix: str
    clip_prefix: str
    cameras: list[str] = Field(min_length=1)
    max_seconds: float | None = Field(None, gt=0)


class DatasetSettings(StrictModel):
    max_height: int = Field(720, ge=144)
    crf: int = Field(23, ge=0, le=51)
    download_confirm_gb: float = Field(20.0, gt=0)
    smartspaces: SmartSpacesSettings = SmartSpacesSettings()
    meva: MevaSettings


class DetectorSettings(StrictModel):
    weights: Path
    imgsz: int = Field(640, ge=64)
    conf: float = Field(0.25, ge=0.0, le=1.0)
    classes: dict[str, int] = Field(min_length=1)
    batch_size: int = Field(4, ge=1)


class TrackerSettings(StrictModel):
    type: Literal["bytetrack"] = "bytetrack"
    track_high_thresh: float = Field(0.25, ge=0.0, le=1.0)
    track_low_thresh: float = Field(0.1, ge=0.0, le=1.0)
    new_track_thresh: float = Field(0.25, ge=0.0, le=1.0)
    track_buffer: int = Field(30, ge=1)
    match_thresh: float = Field(0.8, ge=0.0, le=1.0)
    fuse_score: bool = True


class AnnotateSettings(StrictModel):
    thickness: int = Field(2, ge=1)
    font_scale: float = Field(0.5, gt=0)
    colors: dict[str, tuple[int, int, int]]

    @field_validator("colors")
    @classmethod
    def _roles(cls, v: dict[str, tuple[int, int, int]]) -> dict[str, tuple[int, int, int]]:
        missing = {"resident", "staff", "unknown", "pending", "vehicle"} - set(v)
        if missing:
            raise ValueError(f"annotate.colors missing {sorted(missing)}")
        return v


class PipelineSettings(StrictModel):
    detector: DetectorSettings
    tracker: TrackerSettings = TrackerSettings()
    realtime_max_fps: float = Field(15, gt=0)
    frame_queue_size: int = Field(32, ge=1)
    min_box_height_px: int = Field(24, ge=0)
    engine_autostart: bool = (
        True  # start the engine process with the API (env DWARPAL_ENGINE=0 disables)
    )
    annotate: AnnotateSettings


class ReidSettings(StrictModel):
    backend: Literal["osnet", "colorhist"] = "osnet"
    arch: str = "osnet_x1_0"
    weights: Path
    weights_url: str = ""
    input_height: int = Field(256, ge=32)
    input_width: int = Field(128, ge=16)
    batch_size: int = Field(32, ge=1)
    sample_every: int = Field(5, ge=1)
    max_samples: int = Field(10, ge=1)
    min_quality: float = Field(0.15, ge=0.0, le=1.0)


class GlobalTrackerSettings(StrictModel):
    min_samples: int = Field(3, ge=1)
    match_threshold: float = Field(0.55, ge=-1.0, le=2.0)
    appearance_weight: float = Field(1.0, ge=0.0)
    position_weight: float = Field(0.6, ge=0.0)
    same_place_m: float = Field(1.5, gt=0)
    max_speed_mps: float = Field(3.0, gt=0)
    forget_after_s: float = Field(600, gt=0)
    gallery_size: int = Field(20, ge=1)


class PlateDetectorSettings(StrictModel):
    backend: Literal["oim", "yolo"] = "oim"
    model: str = "yolo-v9-t-640-license-plate-end2end"
    weights: Path | None = None
    conf: float = Field(0.3, ge=0.0, le=1.0)


class PlateOcrSettings(StrictModel):
    model: str = "cct-s-v2-global-model"
    onnx_path: Path | None = None
    config_path: Path | None = None


class AnprSettings(StrictModel):
    vehicle_labels: list[str] = Field(default_factory=lambda: ["car", "motorcycle", "bus", "truck"])
    detector: PlateDetectorSettings = PlateDetectorSettings()
    ocr: PlateOcrSettings = PlateOcrSettings()
    sample_every: int = Field(3, ge=1)
    min_plate_height_px: int = Field(14, ge=1)
    min_char_conf: float = Field(0.4, ge=0.0, le=1.0)
    min_reads: int = Field(3, ge=1)
    min_share: float = Field(0.5, gt=0.0, le=1.0)
    max_reads: int = Field(20, ge=1)
    registry_max_distance: int = Field(1, ge=0)


class ClipSettings(StrictModel):
    model: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    fallback_url: str = ""
    cache_dir: Path = Path("models/clip")
    top_k: int = Field(3, ge=1)


class IndexSettings(StrictModel):
    crops_per_track: int = Field(5, ge=1)
    crop_every: int = Field(5, ge=1)
    min_track_seconds: float = Field(1.0, ge=0)
    max_track_seconds: float = Field(120, gt=0)
    thumb_height: int = Field(192, ge=32)


class Settings(StrictModel):
    app: AppSettings = AppSettings()
    server: ServerSettings = ServerSettings()
    database: DatabaseSettings
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    half_precision: bool = True
    paths: PathSettings
    streaming: StreamingSettings = StreamingSettings()
    identity: IdentitySettings
    privacy: PrivacySettings = PrivacySettings()
    llm: LLMSettings = LLMSettings()
    datasets: DatasetSettings
    pipeline: PipelineSettings
    reid: ReidSettings
    face: FaceSettings = FaceSettings()
    global_tracker: GlobalTrackerSettings = GlobalTrackerSettings()
    anpr: AnprSettings = AnprSettings()
    clip: ClipSettings = ClipSettings()
    index: IndexSettings = IndexSettings()


# --------------------------------------------------------------------------- cameras.yaml


class Zone(StrictModel):
    name: str
    restricted: bool = True
    polygon: list[tuple[float, float]] = Field(min_length=3)

    @field_validator("polygon")
    @classmethod
    def _normalized(cls, poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for x, y in poly:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"zone point {(x, y)} is outside normalized [0, 1] range")
        return poly


class Camera(StrictModel):
    id: str
    name: str
    source_type: Literal["file", "rtsp", "webcam"]
    source_uri: str
    run_mode: Literal["realtime", "cached"] = "realtime"
    enabled: bool = True
    # Processed dataset this camera comes from (GT + calibration under processed_dir/<dataset>/).
    dataset: str | None = None
    anpr: bool = False  # read number plates of vehicles on this camera
    zones: list[Zone] = Field(default_factory=list)
    calibration: dict[str, Any] | None = None

    @field_validator("id")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _SLUG.match(v):
            raise ValueError(f"camera id {v!r} must be a lowercase slug ([a-z0-9_])")
        return v

    @model_validator(mode="after")
    def _check_source(self) -> Camera:
        if self.source_type == "rtsp" and not self.source_uri.startswith(("rtsp://", "rtsps://")):
            raise ValueError(f"camera {self.id}: rtsp source_uri must start with rtsp://")
        if self.source_type == "webcam" and not self.source_uri.isdigit():
            raise ValueError(f"camera {self.id}: webcam source_uri must be a device index")
        names = [z.name for z in self.zones]
        if len(names) != len(set(names)):
            raise ValueError(f"camera {self.id}: duplicate zone names")
        return self


class CamerasConfig(StrictModel):
    cameras: list[Camera] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> CamerasConfig:
        ids = [c.id for c in self.cameras]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate camera ids: {dupes}")
        return self

    def get(self, camera_id: str) -> Camera:
        for cam in self.cameras:
            if cam.id == camera_id:
                return cam
        raise KeyError(camera_id)

    @property
    def zone_names(self) -> set[str]:
        return {z.name for c in self.cameras for z in c.zones}


# --------------------------------------------------------------------------- rules.yaml

RuleType = Literal[
    "unknown_in_zone", "unregistered_vehicle", "loitering", "after_hours", "tailgating"
]


class RuleDefaults(StrictModel):
    cooldown_s: int = Field(300, ge=0)
    presence_gap_s: float = Field(10.0, gt=0)  # unseen this long in a zone = left it


class Rule(StrictModel):
    id: str
    type: RuleType
    enabled: bool = True
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    zones: list[str] = Field(default_factory=list)
    cooldown_s: int | None = Field(None, ge=0)
    params: dict[str, Any] = Field(default_factory=dict)


class RulesConfig(StrictModel):
    defaults: RuleDefaults = RuleDefaults()
    rules: list[Rule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> RulesConfig:
        ids = [r.id for r in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate rule ids")
        return self

    def cooldown_for(self, rule: Rule) -> int:
        return self.defaults.cooldown_s if rule.cooldown_s is None else rule.cooldown_s


# --------------------------------------------------------------------------- loader


class Config(StrictModel):
    root_dir: Path
    config_dir: Path
    settings: Settings
    cameras: CamerasConfig
    rules: RulesConfig

    @model_validator(mode="after")
    def _rule_zones_exist(self) -> Config:
        known = self.cameras.zone_names
        for rule in self.rules.rules:
            missing = [z for z in rule.zones if z not in known]
            if missing:
                raise ValueError(f"rule {rule.id} references unknown zones {missing}")
        return self


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _apply_env_overrides(raw: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    for var, path in ENV_OVERRIDES.items():
        value = env.get(var)
        if not value:
            continue
        node = raw
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return raw


def load_config(
    config_dir: Path | str | None = None, env: Mapping[str, str] | None = None
) -> Config:
    """Load all config files from `config_dir` (default: `$DWARPAL_CONFIG_DIR` or `config/`).

    Pure function of its inputs: pass `env` explicitly in tests.
    """
    env = os.environ if env is None else env
    cdir = Path(config_dir or env.get("DWARPAL_CONFIG_DIR") or DEFAULT_CONFIG_DIR)
    if not cdir.is_absolute():
        cdir = (REPO_ROOT / cdir).resolve()
    root = cdir.parent

    try:
        settings_raw = _apply_env_overrides(_read_yaml(cdir / "settings.yaml"), env)
        settings = Settings.model_validate(settings_raw)
        settings = settings.model_copy(update={"paths": settings.paths.resolved(root)})
        if not settings.clip.cache_dir.is_absolute():
            settings = settings.model_copy(
                update={
                    "clip": settings.clip.model_copy(
                        update={"cache_dir": root / settings.clip.cache_dir}
                    )
                }
            )
        anpr = settings.anpr
        ocr_upd = {
            k: root / v
            for k, v in (("onnx_path", anpr.ocr.onnx_path), ("config_path", anpr.ocr.config_path))
            if v is not None and not v.is_absolute()
        }
        det_upd = (
            {"weights": root / anpr.detector.weights}
            if anpr.detector.weights is not None and not anpr.detector.weights.is_absolute()
            else {}
        )
        if ocr_upd or det_upd:
            anpr = anpr.model_copy(
                update={
                    "ocr": anpr.ocr.model_copy(update=ocr_upd),
                    "detector": anpr.detector.model_copy(update=det_upd),
                }
            )
            settings = settings.model_copy(update={"anpr": anpr})
        if not settings.face.root.is_absolute():
            face = settings.face.model_copy(update={"root": root / settings.face.root})
            settings = settings.model_copy(update={"face": face})
        if not settings.reid.weights.is_absolute():
            reid = settings.reid.model_copy(update={"weights": root / settings.reid.weights})
            settings = settings.model_copy(update={"reid": reid})
        det = settings.pipeline.detector
        if not det.weights.is_absolute():
            det = det.model_copy(update={"weights": root / det.weights})
            pipe = settings.pipeline.model_copy(update={"detector": det})
            settings = settings.model_copy(update={"pipeline": pipe})
        return Config(
            root_dir=root,
            config_dir=cdir,
            settings=settings,
            cameras=CamerasConfig.model_validate(_read_yaml(cdir / "cameras.yaml")),
            rules=RulesConfig.model_validate(_read_yaml(cdir / "rules.yaml")),
        )
    except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
        raise ConfigError(f"invalid config in {cdir}:\n{exc}") from exc


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide config. Reads `.env` (without overriding real env vars) on first call."""
    load_dotenv(REPO_ROOT / ".env", override=False)
    return load_config()
