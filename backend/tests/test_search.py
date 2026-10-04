from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from app.core.config import REPO_ROOT, LLMSettings
from app.db.models import GlobalIdentity, PlateRead, Track, TrackEmbedding
from app.main import create_app
from app.search.parser import QueryParser, RuleParser, SearchFilter
from app.search.retrieval import clip_prompt, search

THUMB = cv2.imencode(
    ".jpg", np.random.default_rng(1).integers(0, 255, (192, 80, 3), dtype=np.uint8)
)[1].tobytes()
CASES = yaml.safe_load((REPO_ROOT / "tests" / "search_queries.yaml").read_text())
NOW = datetime.fromisoformat(CASES["now"])
DEFAULTS = SearchFilter().dump()


def test_at_least_15_queries():
    assert len(CASES["queries"]) >= 15


@pytest.mark.parametrize("case", CASES["queries"], ids=[c["query"] for c in CASES["queries"]])
def test_rule_parser_matches_expected_filter(case, config):
    got = RuleParser(config.cameras, config.settings.app.timezone).parse(case["query"], NOW).dump()
    exp = dict(case["expect"])
    contains = exp.pop("free_text_contains", None)
    check_free = "free_text" in exp
    for key, default in DEFAULTS.items():
        if key == "free_text":
            continue
        want = exp.get(key, default)
        if key == "time_range" and want is not None:
            want = {k: datetime.fromisoformat(v) if v else None for k, v in want.items()}
            have = {
                k: datetime.fromisoformat(v) if v else None for k, v in (got[key] or {}).items()
            }
            assert have == want, (key, got[key])
        elif key == "height_cm" and want is not None:
            assert got[key] == {k: (float(v) if v is not None else None) for k, v in want.items()}
        else:
            assert got[key] == want, (key, got[key])
    if check_free:
        assert got["free_text"] == exp["free_text"]
    if contains:
        assert got["free_text"] and all(w in got["free_text"].split() for w in contains), got[
            "free_text"
        ]


def test_rule_parser_uses_current_time_by_default(config):
    f = RuleParser(config.cameras).parse("unknown people today")
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    assert f.time_range.from_.date() == now.date()


def test_clip_prompt():
    assert clip_prompt("guy green t-shirt") == "a cctv photo of guy green t-shirt"
    assert clip_prompt("backpack") == "a cctv photo of a person, backpack"


# ---------------------------------------------------------------- LLM parser (mocked, no network)


def _llm(provider: str) -> LLMSettings:
    return LLMSettings(provider=provider, model="m", api_key="k")


def test_query_parser_without_provider_uses_rules(config):
    f, by = QueryParser(LLMSettings(), config.cameras).parse("all unknown people today", NOW)
    assert by == "rules" and f.roles == ["unknown"]


def test_anthropic_parser_uses_structured_output(config, monkeypatch):
    import anthropic

    seen = {}
    expected = SearchFilter(roles=["staff"], upper_color="blue", free_text="guard with a lathi")

    class FakeMessages:
        def parse(self, **kw):
            seen.update(kw)
            return SimpleNamespace(stop_reason="end_turn", parsed_output=expected)

    class FakeClient:
        def __init__(self, **kw):
            seen["client"] = kw
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    f, by = QueryParser(_llm("anthropic"), config.cameras).parse("guard in blue with a lathi", NOW)
    assert by == "llm:anthropic" and f == expected
    assert seen["output_format"] is SearchFilter and seen["model"] == "m"
    assert seen["output_config"] == {"effort": "low"}
    assert "guard in blue with a lathi" in seen["messages"][0]["content"]
    assert NOW.isoformat() in seen["messages"][0]["content"]  # time context for "after 9 pm"


def test_anthropic_refusal_or_error_falls_back_to_rules(config, monkeypatch):
    import anthropic

    class Refuses:
        def __init__(self, **kw):
            self.messages = SimpleNamespace(
                parse=lambda **kw: SimpleNamespace(stop_reason="refusal", parsed_output=None)
            )

    monkeypatch.setattr(anthropic, "Anthropic", Refuses)
    f, by = QueryParser(_llm("anthropic"), config.cameras).parse("all unknown people today", NOW)
    assert by == "rules" and f.roles == ["unknown"]

    class Raises:
        def __init__(self, **kw):
            raise RuntimeError("no network")

    monkeypatch.setattr(anthropic, "Anthropic", Raises)
    assert QueryParser(_llm("anthropic"), config.cameras).parse("visitors", NOW)[1] == "rules"


def test_openai_compatible_parser_validates_json(config, monkeypatch):
    import httpx2

    body = {
        "entity": "person",
        "roles": ["unknown"],
        "time_range": {"from": NOW.isoformat(), "to": None},
    }

    class Resp:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": self.content}}]}

    calls = []
    monkeypatch.setattr(
        httpx2, "post", lambda url, **kw: calls.append((url, kw)) or Resp(json.dumps(body))
    )
    f, by = QueryParser(_llm("openai"), config.cameras).parse("strangers since now", NOW)
    assert by == "llm:openai" and f.roles == ["unknown"] and f.time_range.from_ == NOW
    assert calls[0][0] == "https://api.openai.com/v1/chat/completions"
    assert calls[0][1]["json"]["response_format"] == {"type": "json_object"}

    # schema-violating output (unknown colour) -> rules
    monkeypatch.setattr(httpx2, "post", lambda url, **kw: Resp(json.dumps({"upper_color": "teal"})))
    assert QueryParser(_llm("openai"), config.cameras).parse("visitors", NOW)[1] == "rules"


# ---------------------------------------------------------------- retrieval (Postgres)


def unit(*idx: int) -> list[float]:
    v = np.zeros(512, np.float32)
    for i in idx:
        v[i] = 1.0
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def seeded(migrated_db_url, config, tmp_path):
    db = create_engine(migrated_db_url)
    app = create_app(config=config, engine=db, start_engine=False)
    thumb = tmp_path / "t.jpg"
    thumb.write_bytes(THUMB)
    with TestClient(app) as client:  # lifespan syncs cameras
        with Session(db) as s:
            for m in (TrackEmbedding, Track, GlobalIdentity, PlateRead):
                s.execute(delete(m))
            t0 = NOW - timedelta(hours=2)
            s.add_all(
                [
                    GlobalIdentity(id=1, role_state="unknown", first_seen=t0, last_seen=t0),
                    GlobalIdentity(id=2, role_state="resident", first_seen=t0, last_seen=t0),
                ]
            )
            s.flush()
            rows = [  # id, gid, camera, start offset (min), upper, lower, height, zones, clip
                (1, 1, "meva_g336", 0, "green", "blue", 183.0, ["gate_drive"], unit(0)),
                (2, 2, "meva_g336", 10, "green", "black", 165.0, ["gate_drive"], unit(1)),
                (3, 1, "meva_g328", 30, "red", "blue", None, ["resident_parking"], unit(0, 2)),
                (4, None, "meva_g328", 150, "white", "white", 178.0, [], unit(3)),
            ]
            for tid, gid, cam, off, up, lo, h, zones, _clip in rows:
                st = t0 + timedelta(minutes=off)
                s.add(Track(id=tid, global_id=gid, camera_id=cam, start_ts=st, end_ts=st + timedelta(seconds=20),
                            upper_color=up, lower_color=lo, height_cm=h, height_err_cm=4.0 if h else None,
                            zones=zones, role="pending", thumb_path=str(thumb), start_frame=0, end_frame=60))  # fmt: skip
            s.flush()
            for tid, *_, clip in rows:
                s.add(TrackEmbedding(track_id=tid, clip=clip))
            for i, (plate, cam, status) in enumerate(
                [
                    ("TN09AB1234", "meva_g336", "registered"),
                    ("KA01MX5555", "meva_g336", "unregistered"),
                    ("KA01MX5555", "meva_g328", "unregistered"),
                ]
            ):
                s.add(PlateRead(camera_id=cam, ts=NOW - timedelta(minutes=50 - i), plate_text=plate,
                                confidence=0.9, status=status))  # fmt: skip
            s.commit()
        yield client, db
    db.dispose()


@pytest.mark.db
def test_person_filters(seeded):
    _, db = seeded
    with Session(db) as s:
        ids = lambda f: [r["track_id"] for r in search(s, f)["results"]]  # noqa: E731
        assert ids(SearchFilter(roles=["unknown"])) == [3, 1]  # identity role, newest first
        assert ids(SearchFilter(roles=["resident"])) == [2]
        assert ids(SearchFilter(upper_color="green", lower_color="blue")) == [1]
        assert ids(SearchFilter(height_cm={"min": 175, "max": 190})) == [
            4,
            1,
        ]  # null heights excluded
        assert ids(SearchFilter(height_cm={"min": 180})) == [4, 1]  # 178 +- 4 overlaps
        assert ids(SearchFilter(zones=["gate_drive"])) == [2, 1]
        assert ids(SearchFilter(cameras=["meva_g328"])) == [4, 3]
        t0 = NOW - timedelta(hours=2)
        assert ids(
            SearchFilter(
                time_range={"from": t0 + timedelta(minutes=5), "to": t0 + timedelta(minutes=40)}
            )
        ) == [3, 2]


@pytest.mark.db
def test_clip_ranking_and_colour_relaxation(seeded):
    _, db = seeded
    with Session(db) as s:
        out = search(s, SearchFilter(free_text="backpack"), np.array(unit(2)))
        assert out["results"][0]["track_id"] == 3 and out["ranked_by"] == "clip"
        assert out["results"][0]["score"] > out["results"][1]["score"]
        # no purple tracks: colours relaxed, CLIP ranking still answers
        out = search(
            s, SearchFilter(upper_color="purple", free_text="purple top"), np.array(unit(0))
        )
        assert out["relaxed"] == ["upper_color"] and out["results"][0]["track_id"] == 1
        # without free text, a structured miss stays empty
        assert search(s, SearchFilter(upper_color="purple"))["results"] == []


@pytest.mark.db
def test_vehicle_search(seeded):
    _, db = seeded
    with Session(db) as s:
        out = search(s, SearchFilter(entity="vehicle", plate="TN09AB1234"))
        assert out["plate_match"] == "exact" and [r["plate"] for r in out["results"]] == [
            "TN09AB1234"
        ]
        out = search(s, SearchFilter(entity="vehicle", plate="TN09AB1284"))  # one OCR slip
        assert out["plate_match"] == "fuzzy" and out["results"][0]["plate"] == "TN09AB1234"
        assert search(s, SearchFilter(entity="vehicle", plate="DL01ZZ0001"))["results"] == []
        out = search(s, SearchFilter(entity="vehicle", roles=["unknown"]))
        assert {r["status"] for r in out["results"]} == {"unregistered"} and len(
            out["results"]
        ) == 2
        out = search(
            s,
            SearchFilter(entity="vehicle", zones=["gate_drive"]),
            zone_cameras={"gate_drive": ["meva_g336"]},
        )
        assert {r["camera_id"] for r in out["results"]} == {"meva_g336"} and len(
            out["results"]
        ) == 2


@pytest.mark.db
def test_search_api(seeded):
    client, _ = seeded
    client.app.state.text_embedder = lambda text: np.array(unit(0))
    r = client.post(
        "/search",
        json={
            "query": "guy in a green t-shirt, around six foot, near the gate",
            "now": NOW.isoformat(),
        },
    )
    body = r.json()
    assert r.status_code == 200 and body["parsed_by"] == "rules"
    assert body["filter"]["upper_color"] == "green" and body["filter"]["zones"] == ["gate_drive"]
    assert [x["track_id"] for x in body["results"]] == [1]
    assert body["results"][0]["thumb_url"] == "/tracks/1/thumb.jpg"
    r = client.get("/tracks/1/thumb.jpg")  # unknown identity: head blurred when served
    assert r.status_code == 200 and r.content != THUMB
    assert client.get("/tracks/999/thumb.jpg").status_code == 404
    r = client.get("/search", params={"q": "when did TN09AB1234 enter?"})
    assert r.json()["results"][0]["plate"] == "TN09AB1234"
    r = client.post(
        "/search/parse", json={"query": "all unknown people today", "now": NOW.isoformat()}
    )
    assert r.json()["filter"]["roles"] == ["unknown"]


def test_metrics_endpoint_reports_only_saved_results(config, tmp_path):
    import json as _json

    from app.api.metrics import headline

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    h = {m["key"]: m for m in headline(eval_dir)}
    assert h["idf1_multi"]["value"] is None and h["plate_exact"]["value"] is None
    (eval_dir / "plates_synthetic-10.json").write_text(
        _json.dumps(
            {
                "name": "synthetic-10",
                "synthetic": True,
                "ocr_on_crops": {"exact_plate_accuracy": 0.8},
            }
        )
    )
    h = {m["key"]: m for m in headline(eval_dir)}
    assert h["plate_exact"]["value"] is None  # synthetic never becomes the headline plate metric
    assert h["plate_ocr_synthetic"]["value"] == 0.8
