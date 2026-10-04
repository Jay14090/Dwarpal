"""Vehicle registry match: exact -> registered, edit distance <= 1 -> likely_registered."""

from __future__ import annotations

from dataclasses import dataclass


def levenshtein(a: str, b: str, limit: int | None = None) -> int:
    """Edit distance; stops early (returns limit + 1) once it must exceed `limit`."""
    if abs(len(a) - len(b)) > (limit if limit is not None else len(a) + len(b)):
        return (limit or 0) + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if limit is not None and min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class RegistryMatch:
    status: str  # registered | likely_registered | unregistered
    vehicle_id: int | None
    matched_plate: str | None
    distance: int | None


class Registry:
    def __init__(self, plates: dict[str, int] | None = None, max_distance: int = 1) -> None:
        self.plates = dict(plates or {})  # normalized plate -> vehicle id
        self.max_distance = max_distance

    def match(self, plate: str) -> RegistryMatch:
        if plate in self.plates:
            return RegistryMatch("registered", self.plates[plate], plate, 0)
        best: tuple[int, str] | None = None
        for known in self.plates:
            d = levenshtein(plate, known, self.max_distance)
            if d <= self.max_distance and (best is None or (d, known) < best):
                best = (d, known)
        if best is not None:
            return RegistryMatch("likely_registered", self.plates[best[1]], best[1], best[0])
        return RegistryMatch("unregistered", None, None, None)
