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


class IdentitySettings(StrictModel):
    t_face: float = Field(ge=-1.0, le=1.0)
    t_body: float = Field(ge=-1.0, le=1.0)
    min_quality: float = Field(ge=0.0, le=1.0)
    unknown_after_observations: int = Field(ge=1)
    face_weight: float = Field(ge=0.0)
    body_weight: float = Field(ge=0.0)
    staff_vest_fallback: bool = False


class PrivacySettings(StrictModel):
    blur_unknown_faces: bool = True
    unknown_retention_days: int = Field(7, ge=1)
    require_consent: bool = True


class LLMSettings(StrictModel):
    provider: str = ""
    model: str = ""
    api_key: str = Field("", repr=False)
    timeout_s: int = Field(10, ge=1)

    @property
    def enabled(self) -> bool:
        return bool(self.provider and self.model)


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
