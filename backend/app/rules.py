"""Rules engine (CLAUDE.md section 4, P8): person rules evaluated on every processed frame.

Each person is keyed by global id (or camera + local track id while it has none). For every
(person, zone) the engine keeps a *presence*: entered at, last seen. A presence ends once the
person has not been seen in that zone for `gap_s` (detector misses, occlusion). Every rule fires
at most once per presence (one alert per incident, however long the person stays), and the
EventBus cooldown additionally limits each rule to one alert per person per `cooldown_s`, so
walking out and straight back in does not re-alert either.

Rules (config/rules.yaml; `zones: []` means the camera's restricted zones):
- unknown_in_zone: a person whose identity state is in params.roles (default [unknown]).
- loitering: params.roles (default [unknown]) present in the zone >= params.min_dwell_s.
- after_hours: params.roles (default [resident, unknown]: everyone but staff) in the zone during
  [params.start, params.end) local time (window may cross midnight).
- tailgating (stretch): a params.roles person (default [unknown]) enters a zone within
  params.window_s (default 3) after an authorised person (resident/staff) entered it.
`pending` people never trigger rules: the identity state machine resolves them first.
Unregistered vehicles are emitted by the ANPR service through the same EventBus.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, time
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from app.core.config import Camera, Rule, RulesConfig
from app.events import EventBus, EventRecord
from app.pipeline.frames import FrameResult, Track
from app.zones import zones_at

DEFAULT_ROLES = {
    "unknown_in_zone": ["unknown"],
    "loitering": ["unknown"],
    "after_hours": ["resident", "unknown"],
    "tailgating": ["unknown"],
}
AUTHORISED = {"resident", "staff"}


@dataclass
class Presence:
    key: str
    zone: str
    camera_id: str
    entered: float
    last: float
    role: str = "pending"
    fired: set[str] = field(default_factory=set)


def _hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def in_window(t: time, start: time, end: time) -> bool:
    return start <= t < end if start <= end else (t >= start or t < end)


def person_key(camera_id: str, t: Track) -> str:
    return f"g{t.global_id}" if t.global_id is not None else f"{camera_id}:t{t.track_id}"


class RulesEngine:
    def __init__(
        self,
        rules: RulesConfig,
        bus: EventBus,
        tz: str = "Asia/Kolkata",
        gap_s: float = 10.0,
        thumb: Callable[[np.ndarray, tuple[float, float, float, float]], bytes | None]
        | None = None,
    ) -> None:
        self.rules = [r for r in rules.rules if r.enabled and r.type != "unregistered_vehicle"]
        self.bus = bus
        self.tz = ZoneInfo(tz)
        self.gap_s = gap_s
        self.thumb = thumb or crop_jpeg
        self.presence: dict[tuple[str, str], Presence] = {}
        self._lock = threading.Lock()

    def zones_for(self, rule: Rule, camera: Camera) -> set[str]:
        names = {z.name for z in camera.zones}
        return (
            names & set(rule.zones)
            if rule.zones
            else {z.name for z in camera.zones if z.restricted}
        )

    def _expire(self, now: float) -> None:
        for k, p in list(self.presence.items()):
            if now - p.last > self.gap_s:  # left the zone (or lost for good)
                del self.presence[k]

    def observe(self, camera: Camera, result: FrameResult) -> list[EventRecord]:
        ts = result.frame.ts
        img = result.frame.image
        h, w = img.shape[:2]
        fired: list[EventRecord] = []
        with self._lock:
            self._expire(ts)
            entered_now: list[Presence] = []
            seen: list[tuple[Presence, Track]] = []
            for t in result.tracks:
                if not t.is_person:
                    continue
                key = person_key(camera.id, t)
                for zone in zones_at(camera, t.xyxy, w, h):
                    p = self.presence.get((key, zone))
                    if p is None:
                        p = self.presence[(key, zone)] = Presence(key, zone, camera.id, ts, ts)
                        entered_now.append(p)
                    p.last = ts
                    p.camera_id = camera.id
                    p.role = t.role
                    seen.append((p, t))
            local = datetime.fromtimestamp(ts, self.tz).time()
            for p, t in seen:
                for rule in self.rules:
                    if rule.id in p.fired or p.zone not in self.zones_for(rule, camera):
                        continue
                    roles = rule.params.get("roles", DEFAULT_ROLES.get(rule.type, ["unknown"]))
                    if p.role not in roles:
                        continue
                    if self._triggered(rule, p, ts, local, entered_now):
                        fired.extend(self._fire(rule, p, t, result))
        return fired

    def _triggered(
        self, rule: Rule, p: Presence, ts: float, local: time, entered_now: list[Presence]
    ) -> bool:
        if rule.type == "unknown_in_zone":
            return True
        if rule.type == "loitering":
            return ts - p.entered >= float(rule.params.get("min_dwell_s", 60))
        if rule.type == "after_hours":
            return in_window(
                local,
                _hhmm(rule.params.get("start", "22:00")),
                _hhmm(rule.params.get("end", "06:00")),
            )
        if rule.type == "tailgating":
            if p not in entered_now:
                return False
            window = float(rule.params.get("window_s", 3.0))
            return any(
                q.zone == p.zone
                and q.key != p.key
                and q.role in AUTHORISED
                and 0 <= ts - q.entered <= window
                for q in self.presence.values()
            )
        return False

    def _fire(self, rule: Rule, p: Presence, t: Track, result: FrameResult) -> list[EventRecord]:
        p.fired.add(rule.id)  # once per presence, even if the cooldown suppresses it
        payload = {
            "zone": p.zone,
            "role": p.role,
            "track_id": t.track_id,
            "dwell_s": round(p.last - p.entered, 1),
        }
        evs = self.bus.emit(
            rule.type,
            p.key,
            camera_id=p.camera_id,
            ts=result.frame.ts,
            global_id=t.global_id,
            payload=payload,
            rule_filter=lambda r: r.id == rule.id,
            thumb_jpeg=self.thumb(result.frame.image, t.xyxy),
        )
        return evs


def crop_jpeg(
    image: np.ndarray, xyxy: tuple[float, float, float, float], height: int = 192
) -> bytes | None:
    """The person box exactly (privacy.blur_crop blurs its top part when the thumbnail is served)."""
    H, W = image.shape[:2]
    x1, y1, x2, y2 = (
        int(max(0, xyxy[0])),
        int(max(0, xyxy[1])),
        int(min(W, xyxy[2])),
        int(min(H, xyxy[3])),
    )
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    scale = height / crop.shape[0]
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), height))
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


class RulesHook:
    """Per-camera frame hook; runs after IdentityHook so tracks carry their role."""

    def __init__(self, camera: Camera, engine: RulesEngine) -> None:
        self.camera = camera
        self.engine = engine

    def __call__(self, result: FrameResult) -> None:
        self.engine.observe(self.camera, result)
