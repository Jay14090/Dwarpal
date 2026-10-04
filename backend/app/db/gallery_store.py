"""Enrolled people and their embeddings in Postgres (the source of truth for the live gallery).

Consent is enforced here, at the storage layer: nobody is stored without `consent=True`, and
only consenting people are loaded into the gallery (CLAUDE.md privacy rules).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db.models import AuditLog, Person, PersonEmbedding
from app.pipeline.identity import Gallery, GalleryPerson, Kind
from app.pipeline.reid import l2n


class ConsentRequired(ValueError):
    """Enrollment attempted without explicit consent."""


@dataclass
class EnrolledPerson:
    id: int
    role: str
    display_name: str
    unit: str | None
    consent: bool
    face_samples: int
    body_samples: int


def enroll_person(
    session: Session,
    *,
    role: str,
    display_name: str,
    unit: str | None,
    consent: bool,
    face: Sequence[np.ndarray] = (),
    body: Sequence[np.ndarray] = (),
    face_quality: Sequence[float] | None = None,
    body_quality: Sequence[float] | None = None,
    actor: str = "system",
) -> Person:
    if not consent:
        raise ConsentRequired("enrollment requires explicit consent")
    if role not in ("resident", "staff"):
        raise ValueError(f"role must be resident or staff, got {role!r}")
    if not face and not body:
        raise ValueError("enrollment needs at least one face or body embedding")
    person = Person(role=role, display_name=display_name, unit=unit, consent=True)
    session.add(person)
    session.flush()
    for kind, embs, quals in (("face", face, face_quality), ("body", body, body_quality)):
        for i, e in enumerate(embs):
            session.add(
                PersonEmbedding(
                    person_id=person.id,
                    kind=kind,
                    vector=l2n(np.asarray(e, np.float32)),
                    quality=None if quals is None else float(quals[i]),
                )
            )
    session.add(
        AuditLog(
            actor=actor,
            action="enroll",
            target=f"person:{person.id}",
            details={"role": role, "face": len(face), "body": len(body)},
        )
    )
    session.flush()
    return person


def delete_person(session: Session, person_id: int, actor: str = "system") -> bool:
    person = session.get(Person, person_id)
    if person is None:
        return False
    session.delete(person)  # embeddings cascade
    session.add(
        AuditLog(actor=actor, action="delete_person", target=f"person:{person_id}", details={})
    )
    session.flush()
    return True


def delete_people_where_unit(session: Session, unit: str) -> int:
    ids = list(session.scalars(select(Person.id).where(Person.unit == unit)))
    if ids:
        session.execute(delete(Person).where(Person.id.in_(ids)))
    return len(ids)


def list_people(session: Session) -> list[EnrolledPerson]:
    counts: dict[tuple[int, str], int] = {
        (pid, kind): n
        for pid, kind, n in session.execute(
            select(PersonEmbedding.person_id, PersonEmbedding.kind, func.count()).group_by(
                PersonEmbedding.person_id, PersonEmbedding.kind
            )
        )
    }
    return [
        EnrolledPerson(
            p.id,
            p.role,
            p.display_name,
            p.unit,
            p.consent,
            counts.get((p.id, "face"), 0),
            counts.get((p.id, "body"), 0),
        )
        for p in session.scalars(select(Person).order_by(Person.id))
    ]


def load_gallery(session: Session, version: int = 0) -> Gallery:
    people = {
        p.id: GalleryPerson(p.id, p.role, p.display_name)
        for p in session.scalars(select(Person).where(Person.consent.is_(True)))
    }
    embs: dict[Kind, list[tuple[int, np.ndarray]]] = {"face": [], "body": []}
    for pid, kind, vec in session.execute(
        select(PersonEmbedding.person_id, PersonEmbedding.kind, PersonEmbedding.vector)
    ):
        if pid in people:
            embs[kind].append((pid, np.asarray(vec, np.float32)))
    return Gallery(people, embs, version=version)
