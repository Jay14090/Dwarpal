"""Identity engine: gallery of enrolled people + a role state machine per global identity.

Evidence: every face or body sample of a global identity (quality >= min_quality) is matched
against the gallery. A match (cosine >= t_face / t_body) adds `weight * quality` to that
person's score.

    pending  --score(best) >= accept_score------------------------> resident | staff
    pending  --unknown_after_observations samples, no acceptance--> unknown
    unknown  --score(best) >= accept_score (e.g. after enrollment)-> resident | staff
    resident/staff stay put unless another person's score exceeds the current one by
    `switch_ratio` (prevents flicker between similar-looking people)

When the gallery changes (enrollment), every identity's recent samples are re-scored so a
person standing in front of the webcam flips from unknown to resident immediately.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from app.core.config import IdentitySettings

log = logging.getLogger(__name__)

Kind = Literal["face", "body"]
ROLES = ("resident", "staff")


@dataclass
class GalleryPerson:
    person_id: int
    role: str
    display_name: str = ""


class Gallery:
    """Enrolled embeddings as matrices per kind (face/body), rows mapped to person ids."""

    def __init__(
        self,
        people: dict[int, GalleryPerson] | None = None,
        embeddings: dict[Kind, list[tuple[int, np.ndarray]]] | None = None,
        version: int = 0,
    ) -> None:
        self.people = people or {}
        self.version = version
        self._mat: dict[str, np.ndarray] = {}
        self._ids: dict[str, np.ndarray] = {}
        for kind in ("face", "body"):
            rows = (embeddings or {}).get(kind, [])  # type: ignore[call-overload]
            rows = [(pid, e) for pid, e in rows if pid in self.people]
            self._mat[kind] = (
                np.stack([e for _, e in rows]).astype(np.float32)
                if rows
                else np.zeros((0, 512), np.float32)
            )
            self._ids[kind] = np.array([pid for pid, _ in rows], np.int64)

    def __len__(self) -> int:
        return len(self.people)

    def count(self, kind: Kind) -> int:
        return len(self._ids[kind])

    def best_match(self, kind: Kind, emb: np.ndarray) -> tuple[int, float] | None:
        """(person_id, cosine) of the most similar enrolled sample, per person max."""
        m = self._mat[kind]
        if not len(m):
            return None
        sims = m @ emb
        i = int(np.argmax(sims))
        return int(self._ids[kind][i]), float(sims[i])

    def role_of(self, person_id: int) -> str:
        return self.people[person_id].role

    # ------------------------------------------------------------------ persistence

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ids = sorted(self.people)
        np.savez_compressed(
            path,
            person_id=np.array(ids, np.int64),
            role=np.array([self.people[i].role for i in ids]),
            name=np.array([self.people[i].display_name for i in ids]),
            face_ids=self._ids["face"],
            face=self._mat["face"],
            body_ids=self._ids["body"],
            body=self._mat["body"],
        )

    @classmethod
    def load_npz(cls, path: Path) -> Gallery:
        with np.load(path) as z:
            people = {
                int(i): GalleryPerson(int(i), str(r), str(n))
                for i, r, n in zip(z["person_id"], z["role"], z["name"], strict=True)
            }
            embs: dict[Kind, list[tuple[int, np.ndarray]]] = {
                "face": [(int(i), e) for i, e in zip(z["face_ids"], z["face"], strict=True)],
                "body": [(int(i), e) for i, e in zip(z["body_ids"], z["body"], strict=True)],
            }
        return cls(people, embs)


@dataclass
class IdentityState:
    global_id: int
    role_state: str = "pending"
    person_id: int | None = None
    observations: int = 0
    evidence: dict[int, float] = field(default_factory=dict)
    recent: deque[tuple[Kind, np.ndarray, float]] = field(default_factory=lambda: deque(maxlen=20))
    last_change_ts: float = 0.0


Transition = Callable[[IdentityState, str, float], None]  # (state, previous role, ts)


class IdentityEngine:
    def __init__(self, cfg: IdentitySettings, gallery: Gallery | None = None) -> None:
        self.cfg = cfg
        self.gallery = gallery or Gallery()
        self.states: dict[int, IdentityState] = {}
        self.listeners: list[Transition] = []
        self._lock = threading.RLock()

    def state(self, global_id: int) -> IdentityState:
        with self._lock:
            st = self.states.get(global_id)
            if st is None:
                st = IdentityState(global_id, recent=deque(maxlen=self.cfg.recent_samples))
                self.states[global_id] = st
            return st

    def role(self, global_id: int | None) -> str:
        if global_id is None:
            return "pending"
        st = self.states.get(global_id)
        return "pending" if st is None else st.role_state

    def _score(self, st: IdentityState, kind: Kind, emb: np.ndarray, quality: float) -> None:
        match = self.gallery.best_match(kind, emb)
        if match is None:
            return
        pid, sim = match
        threshold = self.cfg.t_face if kind == "face" else self.cfg.t_body
        if sim >= threshold:
            weight = self.cfg.face_weight if kind == "face" else self.cfg.body_weight
            st.evidence[pid] = st.evidence.get(pid, 0.0) + weight * quality

    def _decide(self, st: IdentityState, ts: float) -> None:
        prev = st.role_state
        best = max(st.evidence.items(), key=lambda kv: kv[1], default=None)
        if best is not None and best[1] >= self.cfg.accept_score and best[0] in self.gallery.people:
            pid, score = best
            current = st.evidence.get(st.person_id, 0.0) if st.person_id is not None else 0.0
            if (
                st.person_id is None
                or st.role_state not in ROLES
                or (pid != st.person_id and score >= self.cfg.switch_ratio * current)
            ):
                st.person_id = pid
                st.role_state = self.gallery.role_of(pid)
        elif st.role_state == "pending" and st.observations >= self.cfg.unknown_after_observations:
            st.role_state = "unknown"
        if st.role_state != prev:
            st.last_change_ts = ts
            for cb in self.listeners:
                cb(st, prev, ts)

    def observe(
        self, global_id: int, kind: Kind, emb: np.ndarray, quality: float, ts: float
    ) -> str:
        """Add one face/body sample to a global identity; returns its role state."""
        if quality < self.cfg.min_quality:
            return self.role(global_id)
        with self._lock:
            st = self.state(global_id)
            st.observations += 1
            st.recent.append((kind, emb, quality))
            self._score(st, kind, emb, quality)
            self._decide(st, ts)
            return st.role_state

    def set_gallery(self, gallery: Gallery, ts: float) -> None:
        """Swap the gallery and re-score every identity's recent samples against it."""
        with self._lock:
            self.gallery = gallery
            for st in self.states.values():
                st.evidence.clear()
                if st.person_id is not None and st.person_id not in gallery.people:
                    st.person_id = None  # person was deleted
                    st.role_state = "pending"
                for kind, emb, q in st.recent:
                    self._score(st, kind, emb, q)
                self._decide(st, ts)
            log.info("gallery v%d: %d people (%d face, %d body samples)", gallery.version,
                     len(gallery), gallery.count("face"), gallery.count("body"))  # fmt: skip

    def forget(self, global_id: int) -> None:
        with self._lock:
            self.states.pop(global_id, None)
