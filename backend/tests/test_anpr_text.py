from __future__ import annotations

import pytest

from app.anpr.normalize import BH_RE, STANDARD_RE, normalize
from app.anpr.registry import Registry, levenshtein


@pytest.mark.parametrize(
    ("raw", "text", "kind", "corrections"),
    [
        ("TN09AB1234", "TN09AB1234", "standard", 0),
        ("tn 09 ab 1234", "TN09AB1234", "standard", 0),
        ("TN-O9-AB-1234", "TN09AB1234", "standard", 1),  # O read for 0 in the district
        ("MH12AB1234XYZ", "MH12AB1234XYZ", "invalid", 0),  # too long for any format
        ("DL1CAB1234", "DL1CAB1234", "standard", 0),  # 1-digit district, 3-letter series
        ("KA05MJ8B21", "KA05MJ8821", "standard", 1),  # B read for 8 in the number
        ("8H12AB1234", "BH12AB1234", "standard", 1),
        ("22BH1234AB", "22BH1234AB", "bh", 0),
        ("MH 12 A 1234", "MH12A1234", "standard", 0),  # 1-letter series
        ("22 8H 1234 A", "22BH1234A", "bh", 1),
        ("HELLO", "HELLO", "invalid", 0),
        ("", "", "invalid", 0),
    ],
)
def test_normalize(raw, text, kind, corrections):
    p = normalize(raw)
    assert (p.text, p.kind, p.corrections) == (text, kind, corrections)
    if p.valid:
        assert (STANDARD_RE if kind == "standard" else BH_RE).match(p.text)


def test_prefers_valid_state_codes():
    # "0L" could be letters O-L (not a state) or... only letters fit; check the state flag
    assert normalize("TN09AB1234").valid_state
    assert not normalize("ZZ09AB1234").valid_state


def test_levenshtein():
    assert levenshtein("TN09AB1234", "TN09AB1234") == 0
    assert levenshtein("TN09AB1234", "TN09AB1235") == 1
    assert levenshtein("TN09AB1234", "TN9AB1234") == 1
    assert levenshtein("TN09AB1234", "KA01ZZ9999", limit=1) == 2


def test_registry_statuses():
    reg = Registry({"TN09AB1234": 1, "KA05MJ8821": 2})
    assert reg.match("TN09AB1234").status == "registered"
    m = reg.match("TN09AB1284")
    assert (m.status, m.vehicle_id, m.matched_plate) == ("likely_registered", 1, "TN09AB1234")
    assert reg.match("MH12AB1234").status == "unregistered"
    assert reg.match("MH12AB1234").vehicle_id is None
