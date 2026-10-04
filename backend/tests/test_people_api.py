from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.models import AuditLog, Person, PersonEmbedding
from app.main import create_app

pytestmark = pytest.mark.db
JPEG = b"\xff\xd8thumb\xff\xd9"


class FakeEngineProc:
    alive = True

    def __init__(self, n_face=4, n_body=4):
        self.calls: list[tuple[str, dict]] = []
        self.n_face, self.n_body = n_face, n_body

    def request(self, cmd, timeout, **args):
        self.calls.append((cmd, args))
        if cmd == "reload_gallery":
            return {"people": 1}
        if cmd in ("enroll_capture", "embed_images"):
            v = (np.ones(512) / np.sqrt(512)).tolist()
            return {"face": [v] * self.n_face, "face_quality": [0.9] * self.n_face,
                    "body": [v] * self.n_body, "body_quality": [0.5] * self.n_body,
                    "thumb_jpeg": JPEG, "frames_with_person": 20}  # fmt: skip
        return {"error": "?"}


@pytest.fixture
def api(migrated_db_url, config, tmp_path):
    paths = config.settings.paths.model_copy(update={"thumbs_dir": tmp_path / "thumbs"})
    cfg = config.model_copy(
        update={"settings": config.settings.model_copy(update={"paths": paths})}
    )
    db = create_engine(migrated_db_url)
    app = create_app(config=cfg, engine=db, start_engine=False)
    app.state.engine_proc = FakeEngineProc()
    with Session(db) as s:  # isolate from other tests sharing the database
        s.query(Person).delete()
        s.query(AuditLog).delete()
        s.commit()
    yield TestClient(app), app, db
    db.dispose()


BODY = {"role": "resident", "display_name": "Asha", "unit": "B-204", "consent": True, "seconds": 1}


def test_capture_requires_consent(api):
    client, app, db = api
    r = client.post("/enroll/capture", json={**BODY, "consent": False})
    assert r.status_code == 422 and "consent" in r.text
    assert app.state.engine_proc.calls == []  # nothing was captured
    with Session(db) as s:
        assert s.scalar(select(Person)) is None


def test_capture_enrolls_person_with_embeddings_thumb_and_audit(api):
    client, app, db = api
    r = client.post("/enroll/capture", json=BODY, headers={"X-Actor": "guard-1"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["face_shots"] == 4 and out["body_shots"] == 4 and out["role"] == "resident"
    assert [c for c, _ in app.state.engine_proc.calls] == ["enroll_capture", "reload_gallery"]
    with Session(db) as s:
        p = s.get(Person, out["id"])
        assert p.consent and p.unit == "B-204"
        kinds = sorted(
            k
            for (k,) in s.execute(
                select(PersonEmbedding.kind).where(PersonEmbedding.person_id == p.id)
            )
        )
        assert kinds == ["body"] * 4 + ["face"] * 4
        audit = s.scalars(select(AuditLog)).all()
        assert [(a.actor, a.action) for a in audit] == [("guard-1", "enroll")]
    assert client.get(f"/people/{out['id']}/thumb.jpg").content == JPEG
    assert client.get("/people").json()[0]["face_samples"] == 4


def test_too_few_good_shots_is_rejected_with_a_hint(api):
    client, app, _ = api
    app.state.engine_proc = FakeEngineProc(n_face=1, n_body=2)
    r = client.post("/enroll/capture", json=BODY)
    assert r.status_code == 422 and "stand closer" in r.text


def test_upload_enrollment(api):
    client, _, _ = api
    files = [("images", (f"{i}.jpg", JPEG, "image/jpeg")) for i in range(3)]
    data = {"role": "staff", "display_name": "Ravi", "consent": "true"}
    r = client.post("/enroll/upload", data=data, files=files)
    assert r.status_code == 200, r.text
    assert r.json()["role"] == "staff"
    r = client.post("/enroll/upload", data={**data, "consent": "false"}, files=files)
    assert r.status_code == 422


def test_delete_person_is_audited(api):
    client, _, db = api
    pid = client.post("/enroll/capture", json=BODY).json()["id"]
    assert client.delete(f"/people/{pid}", headers={"X-Actor": "admin"}).status_code == 200
    assert client.delete(f"/people/{pid}").status_code == 404
    with Session(db) as s:
        assert s.get(Person, pid) is None
        assert s.scalar(select(PersonEmbedding).where(PersonEmbedding.person_id == pid)) is None
        assert ("admin", "delete_person") in [
            (a.actor, a.action) for a in s.scalars(select(AuditLog))
        ]


def test_enroll_without_engine_is_503(api):
    client, app, _ = api
    app.state.engine_proc = None
    assert client.post("/enroll/capture", json=BODY).status_code == 503


def test_gallery_only_loads_consenting_people(api):
    from app.db.gallery_store import ConsentRequired, enroll_person, load_gallery

    _, _, db = api
    v = np.ones(512) / np.sqrt(512)
    with Session(db) as s:
        ok = enroll_person(s, role="resident", display_name="A", unit=None, consent=True, face=[v])
        # bypass the store to simulate a legacy/bad row without consent
        bad = Person(role="staff", display_name="B", consent=False)
        s.add(bad)
        s.flush()
        s.add(PersonEmbedding(person_id=bad.id, kind="face", vector=v))
        with pytest.raises(ConsentRequired):
            enroll_person(s, role="resident", display_name="C", unit=None, consent=False, face=[v])
        s.commit()
        g = load_gallery(s)
    assert set(g.people) == {ok.id}
    assert g.count("face") == 1
