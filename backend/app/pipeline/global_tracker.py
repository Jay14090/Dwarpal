"""Cross-camera global identities: appearance (Re-ID) + spatio-temporal gating.

A *sighting* is one local track in one camera. It collects its best-quality embeddings and, on
calibrated cameras, its ground-plane positions. Once it has `min_samples` embeddings it is
matched to an existing global identity or starts a new one:

  infeasible  - the identity is live on another track in the same camera (one body, one track)
              - the identity is live in another camera and both are calibrated, but the two
                ground positions differ by more than `same_place_m`
              - the identity was last seen at a place that needs > `max_speed_mps` to reach
  score       = appearance_weight * cos_sim(sighting, identity gallery)
              + position_weight * exp(-(d / same_place_m)^2)   (only for concurrent sightings)

The best feasible identity with score >= match_threshold wins. Thread-safe: all cameras of
the engine share one instance.
"""

from __future__ import annotations

import itertools
import math
import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from app.core.config import GlobalTrackerSettings
from app.pipeline.reid import l2n

LIVE_GAP_S = 1.0  # a sighting counts as "live" if it was observed within this many seconds


@dataclass
class Sighting:
    camera_id: str
    track_id: int
    first_ts: float
    last_ts: float
    max_samples: int
    samples: list[tuple[float, np.ndarray]] = field(default_factory=list)  # (quality, emb)
    positions: deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=50))
    global_id: int | None = None
    ended: bool = False

    def add_sample(self, quality: float, emb: np.ndarray) -> None:
        self.samples.append((quality, emb))
        if len(self.samples) > self.max_samples:
            self.samples.sort(key=lambda s: s[0], reverse=True)
            self.samples.pop()

    @property
    def embedding(self) -> np.ndarray | None:
        if not self.samples:
            return None
        w = np.array([q for q, _ in self.samples], np.float32)[:, None]
        e = np.stack([e for _, e in self.samples])
        return l2n((e * np.maximum(w, 1e-3)).sum(0))

    def position_near(self, ts: float, max_dt: float = LIVE_GAP_S) -> tuple[float, float] | None:
        best = None
        for t, x, y in self.positions:
            dt = abs(t - ts)
            if dt <= max_dt and (best is None or dt < best[0]):
                best = (dt, x, y)
        return None if best is None else (best[1], best[2])

    def live_at(self, ts: float) -> bool:
        return not self.ended and ts - self.last_ts <= LIVE_GAP_S


@dataclass
class GlobalIdentity:
    id: int
    gallery: deque[np.ndarray]
    first_seen: float
    last_seen: float
    sightings: dict[tuple[str, int], Sighting] = field(default_factory=dict)
    last_position: tuple[float, float, float] | None = None  # (ts, x, y)

    def similarity(self, emb: np.ndarray) -> float:
        if not self.gallery:
            return -1.0
        sims = np.stack(list(self.gallery)) @ emb
        top = np.sort(sims)[-3:]
        return float(top.mean())


class GlobalTracker:
    def __init__(
        self, cfg: GlobalTrackerSettings, max_samples: int = 10, start_id: int = 1
    ) -> None:
        """`start_id`: first id to hand out (the engine continues after ids already in the DB)."""
        self.cfg = cfg
        self.max_samples = max_samples
        self._lock = threading.RLock()
        self._ids = itertools.count(start_id)
        self.identities: dict[int, GlobalIdentity] = {}
        self.sightings: dict[tuple[str, int], Sighting] = {}
        self.new_identity_listeners: list = []

    # ------------------------------------------------------------------ updates

    def observe(
        self,
        camera_id: str,
        track_id: int,
        ts: float,
        emb: np.ndarray | None = None,
        quality: float = 0.0,
        world_xy: tuple[float, float] | None = None,
    ) -> int | None:
        """Record one frame of a person track. Returns its global id (None while pending)."""
        key = (camera_id, track_id)
        with self._lock:
            s = self.sightings.get(key)
            if s is None:
                s = Sighting(camera_id, track_id, ts, ts, self.max_samples)
                self.sightings[key] = s
            s.last_ts = ts
            if emb is not None:
                s.add_sample(quality, emb)
            if world_xy is not None:
                s.positions.append((ts, float(world_xy[0]), float(world_xy[1])))
            if s.global_id is None and len(s.samples) >= self.cfg.min_samples:
                self._assign(s, ts)
            if s.global_id is not None:
                ident = self.identities[s.global_id]
                ident.last_seen = max(ident.last_seen, ts)
                if world_xy is not None:
                    ident.last_position = (ts, float(world_xy[0]), float(world_xy[1]))
            return s.global_id

    def end(self, camera_id: str, track_id: int) -> None:
        """Local track finished: fold its appearance into the identity gallery."""
        with self._lock:
            s = self.sightings.pop((camera_id, track_id), None)
            if s is None:
                return
            s.ended = True
            if s.global_id is None:
                return
            ident = self.identities[s.global_id]
            ident.sightings.pop((camera_id, track_id), None)
            emb = s.embedding
            if emb is not None:
                ident.gallery.append(emb)

    def global_id(self, camera_id: str, track_id: int) -> int | None:
        s = self.sightings.get((camera_id, track_id))
        return None if s is None else s.global_id

    # ------------------------------------------------------------------ matching

    def _feasible_score(
        self, s: Sighting, ident: GlobalIdentity, emb: np.ndarray, ts: float
    ) -> float | None:
        cfg = self.cfg
        pos_bonus = 0.0
        my_pos = s.position_near(ts)
        for other in ident.sightings.values():
            if other is s or not other.live_at(ts):
                continue
            if other.camera_id == s.camera_id:
                return None  # already a live track in this camera
            other_pos = other.position_near(ts)
            if my_pos is not None and other_pos is not None:
                d = math.dist(my_pos, other_pos)
                if d > cfg.same_place_m:
                    return None
                pos_bonus = max(
                    pos_bonus, cfg.position_weight * math.exp(-((d / cfg.same_place_m) ** 2))
                )
        if pos_bonus == 0.0 and ident.last_position is not None and s.positions:
            t0, x0, y0 = ident.last_position
            t1, x1, y1 = s.positions[0]
            gap = abs(t1 - t0)
            if gap > LIVE_GAP_S and math.dist((x0, y0), (x1, y1)) / gap > cfg.max_speed_mps:
                return None
        return cfg.appearance_weight * ident.similarity(emb) + pos_bonus

    def _assign(self, s: Sighting, ts: float) -> None:
        emb = s.embedding
        assert emb is not None
        best: tuple[float, GlobalIdentity] | None = None
        for ident in self.identities.values():
            if ts - ident.last_seen > self.cfg.forget_after_s and not ident.sightings:
                continue
            score = self._feasible_score(s, ident, emb, ts)
            if (
                score is not None
                and score >= self.cfg.match_threshold
                and (best is None or score > best[0])
            ):
                best = (score, ident)
        if best is None:
            ident = GlobalIdentity(
                id=next(self._ids),
                gallery=deque([emb], maxlen=self.cfg.gallery_size),
                first_seen=s.first_ts,
                last_seen=ts,
            )
            self.identities[ident.id] = ident
            for cb in self.new_identity_listeners:
                cb(ident)
        else:
            ident = best[1]
            ident.gallery.append(emb)
        s.global_id = ident.id
        ident.sightings[(s.camera_id, s.track_id)] = s

    # ------------------------------------------------------------------ housekeeping

    def end_stale(
        self, camera_id: str, live_track_ids: set[int], ts: float, grace_s: float
    ) -> list[int]:
        """End sightings of `camera_id` not observed for `grace_s`. Returns ended track ids."""
        with self._lock:
            stale = [
                tid
                for (cam, tid), s in self.sightings.items()
                if cam == camera_id and tid not in live_track_ids and ts - s.last_ts > grace_s
            ]
        for tid in stale:
            self.end(camera_id, tid)
        return stale

    def reset(self) -> None:
        with self._lock:
            self.identities.clear()
            self.sightings.clear()
            self._ids = itertools.count(1)
