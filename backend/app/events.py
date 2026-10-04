"""Event emission with per-rule cooldown (dedup) shared by every producer.

Producers (plate reads in P5, the rules engine in P8) call `EventBus.emit(rule_type, key, ...)`.
Every enabled rule of that type in config/rules.yaml fires at most once per `key` (a global id,
a plate) per cooldown window; the sink persists and publishes the event.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.config import Rule, RulesConfig


@dataclass
class EventRecord:
    rule: str  # rule id from rules.yaml
    rule_type: str
    severity: str
    camera_id: str | None
    ts: float
    global_id: int | None = None
    plate_read_id: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    id: int | None = None  # set once stored

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


Sink = Callable[[EventRecord], None]


class EventBus:
    def __init__(self, rules: RulesConfig, sink: Sink) -> None:
        self.rules = rules
        self.sink = sink
        self._last: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def rules_of(self, rule_type: str) -> list[Rule]:
        return [r for r in self.rules.rules if r.enabled and r.type == rule_type]

    def allow(self, rule: Rule, key: str, ts: float) -> bool:
        """Cooldown check-and-set for (rule, key)."""
        with self._lock:
            last = self._last.get((rule.id, key))
            if last is not None and ts - last < self.rules.cooldown_for(rule):
                return False
            self._last[(rule.id, key)] = ts
            return True

    def emit(
        self,
        rule_type: str,
        key: str,
        *,
        camera_id: str | None,
        ts: float | None = None,
        global_id: int | None = None,
        plate_read_id: int | None = None,
        payload: dict[str, Any] | None = None,
        rule_filter: Callable[[Rule], bool] | None = None,
    ) -> list[EventRecord]:
        ts = time.time() if ts is None else ts
        out = []
        for rule in self.rules_of(rule_type):
            if rule_filter is not None and not rule_filter(rule):
                continue
            if not self.allow(rule, key, ts):
                continue
            ev = EventRecord(
                rule.id,
                rule.type,
                rule.severity,
                camera_id,
                ts,
                global_id,
                plate_read_id,
                dict(payload or {}),
            )
            self.sink(ev)
            out.append(ev)
        return out
