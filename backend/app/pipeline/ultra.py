"""Ultralytics import wrapper that turns off its usage analytics (privacy: CLAUDE.md section 4,
nothing leaves the machine). Import YOLO / trackers through here."""

from __future__ import annotations

from typing import Any


def disable_telemetry() -> None:
    from ultralytics.utils import SETTINGS

    if SETTINGS.get("sync", False):
        SETTINGS.update({"sync": False})  # persisted in Ultralytics' settings.json
    from ultralytics.utils import events as events_mod

    events_mod.events.enabled = False  # instance was created before the setting changed


def yolo(weights: str) -> Any:
    disable_telemetry()
    from ultralytics import YOLO

    return YOLO(weights)
