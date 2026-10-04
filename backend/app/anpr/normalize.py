"""Normalize OCR output to Indian registration formats (CLAUDE.md section 4, ANPR).

Formats
  standard  ^[A-Z]{2}\\d{1,2}[A-Z]{0,3}\\d{4}$   e.g. TN09AB1234, DL1CAB1234, MH12A1234
  bh        ^\\d{2}BH\\d{4}[A-Z]{1,2}$           e.g. 22BH1234AB (Bharat series)

OCR confuses letters and digits (O/0, I/1, B/8, ...). We try every way the string can fit a
template (district digits 1-2, series letters 0-3), and at each position accept the character
if it matches the expected type, or swap it for its confusable counterpart at a cost of one.
The cheapest fit wins; ties prefer a valid state code. No fit means `kind="invalid"`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

STANDARD_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$")
BH_RE = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")

# RTO state / UT codes (current and legacy codes still on the road)
STATE_CODES = frozenset(
    [
        "AN",
        "AP",
        "AR",
        "AS",
        "BR",
        "CG",
        "CH",
        "CT",
        "DD",
        "DL",
        "DN",
        "GA",
        "GJ",
        "HP",
        "HR",
        "JH",
        "JK",
        "KA",
        "KL",
        "LA",
        "LD",
        "MH",
        "ML",
        "MN",
        "MP",
        "MZ",
        "NL",
        "OD",
        "OR",
        "PB",
        "PY",
        "RJ",
        "SK",
        "TG",
        "TN",
        "TR",
        "TS",
        "UK",
        "UP",
        "UT",
        "WB",
    ]
)
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "U": "0", "I": "1", "L": "1", "J": "1", "Z": "2",
            "A": "4", "S": "5", "G": "6", "T": "7", "B": "8"}  # fmt: skip


@dataclass(frozen=True)
class NormalizedPlate:
    text: str  # corrected plate, or the cleaned raw string when invalid
    raw: str
    kind: str  # standard | bh | invalid
    corrections: int
    valid_state: bool

    @property
    def valid(self) -> bool:
        return self.kind != "invalid"


def clean(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def _fit(s: str, template: str) -> tuple[str, int] | None:
    """template: 'L' letter, 'D' digit, or a literal letter (e.g. BH). Returns (fixed, cost)."""
    if len(s) != len(template):
        return None
    out, cost = [], 0
    for ch, t in zip(s, template, strict=True):
        if t == "L":
            if ch.isalpha():
                out.append(ch)
            elif ch in TO_LETTER:
                out.append(TO_LETTER[ch])
                cost += 1
            else:
                return None
        elif t == "D":
            if ch.isdigit():
                out.append(ch)
            elif ch in TO_DIGIT:
                out.append(TO_DIGIT[ch])
                cost += 1
            else:
                return None
        else:  # literal
            if ch == t:
                out.append(ch)
            elif TO_LETTER.get(ch) == t:
                out.append(t)
                cost += 1
            else:
                return None
    return "".join(out), cost


def _templates(n: int) -> list[tuple[str, str]]:
    temps = []
    for district in (1, 2):
        for series in range(4):
            if 2 + district + series + 4 == n:
                temps.append(("standard", "LL" + "D" * district + "L" * series + "DDDD"))
    for tail in (1, 2):
        if 8 + tail == n:
            temps.append(("bh", "DDBHDDDD" + "L" * tail))
    return temps


def normalize(raw: str) -> NormalizedPlate:
    s = clean(raw)
    best: tuple[int, int, str, str] | None = None  # (cost, state_penalty, text, kind)
    for kind, template in _templates(len(s)):
        fit = _fit(s, template)
        if fit is None:
            continue
        text, cost = fit
        state_ok = kind == "bh" or text[:2] in STATE_CODES
        key = (cost, 0 if state_ok else 1, text, kind)
        if best is None or key[:2] < best[:2]:
            best = key
    if best is None:
        return NormalizedPlate(s, raw, "invalid", 0, False)
    cost, penalty, text, kind = best
    pattern = STANDARD_RE if kind == "standard" else BH_RE
    assert pattern.match(text), text
    return NormalizedPlate(text, raw, kind, cost, penalty == 0)
