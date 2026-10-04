from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.core.config import ConfigError, load_config


def _edit(path: Path, mutate) -> None:
    data = yaml.safe_load(path.read_text())
    mutate(data)
    path.write_text(yaml.safe_dump(data))


def test_shipped_config_loads(config):
    s = config.settings
    assert s.device in {"auto", "cuda", "mps", "cpu"}
    assert s.database.url.startswith("postgresql")
    assert s.privacy.unknown_retention_days == 7
    assert s.paths.data_dir.is_absolute()
    assert config.cameras.get("webcam").run_mode == "realtime"
    assert {r.type for r in config.rules.rules} >= {
        "unknown_in_zone",
        "unregistered_vehicle",
        "loitering",
        "after_hours",
    }


def test_env_overrides(config_dir):
    cfg = load_config(
        config_dir,
        env={
            "DATABASE_URL": "postgresql+psycopg://u:p@db:5432/x",
            "DWARPAL_DEVICE": "cpu",
            "LLM_PROVIDER": "anthropic",
            "LLM_MODEL": "some-model",
            "LLM_API_KEY": "secret",
        },
    )
    assert cfg.settings.database.url == "postgresql+psycopg://u:p@db:5432/x"
    assert cfg.settings.device == "cpu"
    assert cfg.settings.llm.enabled
    assert "secret" not in repr(cfg.settings.llm)


def test_llm_disabled_without_env(config):
    assert not config.settings.llm.enabled


def test_unknown_key_rejected(config_dir):
    _edit(config_dir / "settings.yaml", lambda d: d["identity"].update(t_fcae=0.5))
    with pytest.raises(ConfigError, match="t_fcae"):
        load_config(config_dir, env={})


def test_invalid_device_rejected(config_dir):
    with pytest.raises(ConfigError, match="device"):
        load_config(config_dir, env={"DWARPAL_DEVICE": "tpu"})


def test_duplicate_camera_ids_rejected(config_dir):
    _edit(config_dir / "cameras.yaml", lambda d: d["cameras"].append(dict(d["cameras"][0])))
    with pytest.raises(ConfigError, match="duplicate camera ids"):
        load_config(config_dir, env={})


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_type", "ftp", "source_type"),
        ("source_uri", "http://not-rtsp", "rtsp://"),
        ("id", "Gate Cam", "slug"),
    ],
)
def test_bad_camera_fields_rejected(config_dir, field, value, match):
    _edit(config_dir / "cameras.yaml", lambda d: d["cameras"][0].update({field: value}))
    with pytest.raises(ConfigError, match=match):
        load_config(config_dir, env={})


def test_webcam_source_needs_device_index(config_dir):
    _edit(
        config_dir / "cameras.yaml",
        lambda d: d["cameras"][0].update(source_type="webcam", source_uri="/dev/video0"),
    )
    with pytest.raises(ConfigError, match="device index"):
        load_config(config_dir, env={})


def test_zone_must_be_normalized(config_dir):
    _edit(
        config_dir / "cameras.yaml",
        lambda d: d["cameras"][0]["zones"][0].update(polygon=[[0, 0], [640, 0], [640, 480]]),
    )
    with pytest.raises(ConfigError, match="normalized"):
        load_config(config_dir, env={})


def test_zone_needs_three_points(config_dir):
    _edit(
        config_dir / "cameras.yaml",
        lambda d: d["cameras"][0]["zones"][0].update(polygon=[[0, 0], [1, 1]]),
    )
    with pytest.raises(ConfigError, match="polygon"):
        load_config(config_dir, env={})


def test_rule_zone_must_exist(config_dir):
    _edit(config_dir / "rules.yaml", lambda d: d["rules"][0].update(zones=["nowhere"]))
    with pytest.raises(ConfigError, match="unknown zones"):
        load_config(config_dir, env={})


def test_rule_cooldown_falls_back_to_default(config):
    rules = config.rules
    assert all(rules.cooldown_for(r) == rules.defaults.cooldown_s for r in rules.rules)


def test_missing_file_is_a_config_error(config_dir):
    (config_dir / "rules.yaml").unlink()
    with pytest.raises(ConfigError, match=r"missing config file.*rules\.yaml"):
        load_config(config_dir, env={})
