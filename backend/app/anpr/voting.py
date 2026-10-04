"""Multi-frame plate voting per vehicle track.

Each valid read votes for its normalized text with weight = OCR confidence. A plate is decided
once the leader has `min_reads` reads and `min_share` of the total weight; when the track ends
the best plate so far is decided if it has at least one valid read. Reads that fail the Indian
format never win against valid ones.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from app.anpr.normalize import NormalizedPlate


@dataclass
class PlateVote:
    plate: str
    confidence: float  # leader's share of the vote weight, times its mean OCR confidence
    reads: int
    total_reads: int
    raw_examples: list[str]


@dataclass
class TrackVotes:
    reads: list[tuple[NormalizedPlate, float]] = field(default_factory=list)
    decided: PlateVote | None = None

    def tally(self) -> PlateVote | None:
        weights: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        raws: dict[str, list[str]] = defaultdict(list)
        for p, conf in self.reads:
            if not p.valid:
                continue
            weights[p.text] += conf
            counts[p.text] += 1
            raws[p.text].append(p.raw)
        if not weights:
            return None
        leader = max(weights, key=lambda t: (weights[t], counts[t]))
        share = weights[leader] / sum(weights.values())
        mean_conf = weights[leader] / counts[leader]
        return PlateVote(
            leader, share * mean_conf, counts[leader], len(self.reads), raws[leader][:5]
        )


class PlateVoter:
    def __init__(self, min_reads: int, min_share: float, max_reads: int) -> None:
        self.min_reads = min_reads
        self.min_share = min_share
        self.max_reads = max_reads
        self.tracks: dict[int, TrackVotes] = {}

    def add(self, track_id: int, plate: NormalizedPlate, conf: float) -> PlateVote | None:
        """Add a read; returns the decision the first time the track's plate is decided."""
        tv = self.tracks.setdefault(track_id, TrackVotes())
        if tv.decided is not None or len(tv.reads) >= self.max_reads:
            return None
        tv.reads.append((plate, conf))
        vote = tv.tally()
        if vote is None or vote.reads < self.min_reads:
            return None
        weights = sum(c for p, c in tv.reads if p.valid)
        share = sum(c for p, c in tv.reads if p.valid and p.text == vote.plate) / weights
        if share >= self.min_share:
            tv.decided = vote
            return vote
        return None

    def finish(self, track_id: int) -> PlateVote | None:
        """Track ended: decide on the best valid plate if not decided yet."""
        tv = self.tracks.pop(track_id, None)
        if tv is None or tv.decided is not None:
            return None
        return tv.tally()

    def decided(self, track_id: int) -> PlateVote | None:
        tv = self.tracks.get(track_id)
        return None if tv is None else tv.decided
