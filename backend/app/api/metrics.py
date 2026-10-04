"""Metrics page data: the reports evaluation scripts wrote to data/eval/ (never computed or invented here).

Each headline metric points at the report it came from; a metric with no report is returned with
`value: null` and the command that measures it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(tags=["metrics"])


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def headline(eval_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add(
        key: str, label: str, value: float | None, source: str | None, how: str, detail: str = ""
    ) -> None:
        out.append(
            {
                "key": key,
                "label": label,
                "value": value,
                "source": source,
                "how": how,
                "detail": detail,
            }
        )

    mtmc = sorted(eval_dir.glob("*_mtmc.json"))
    rep = _load(mtmc[-1]) if mtmc else None
    run = rep["runs"][0] if rep and rep.get("runs") else None
    add("idf1_multi", "Cross-camera IDF1", run["multi_camera_online"]["idf1"] if run else None,
        mtmc[-1].name if run else None, "make eval",
        f"{rep['dataset']}, {len(rep['cameras'])} cameras, frames {run['frames']}" if run else "")  # fmt: skip
    add("idf1_single", "Single-camera IDF1", run["single_camera"]["idf1"] if run else None,
        mtmc[-1].name if run else None, "make eval")  # fmt: skip

    ident = sorted(eval_dir.glob("*_identity.json"))
    r = _load(ident[-1]) if ident else None
    src = ident[-1].name if r else None
    add("role_accuracy", "Role-label accuracy", r["role_accuracy"] if r else None, src, "make enroll-sim && make eval",
        f"{r['decided_obs']} decided observations, held-out frames {r['frames']}" if r else "")  # fmt: skip
    add("unknown_precision", "Unknown-alert precision", r["unknown_precision"] if r else None, src,
        "make enroll-sim && make eval", f"{r['alerts_correct']}/{r['alerts']} alerts" if r else "")  # fmt: skip
    add("unknown_recall", "Unknown-alert recall", r["unknown_recall"] if r else None, src,
        "make enroll-sim && make eval", f"{r['unknown_people_alerted']}/{r['unknown_people']} unknown people" if r else "")  # fmt: skip

    plates = [(p, _load(p)) for p in sorted(eval_dir.glob("plates_*.json"))]
    real = [(p, d) for p, d in plates if d and not d.get("synthetic") and "end_to_end" in d]
    if real:
        p, d = real[-1]
        add("plate_exact", "Exact-plate accuracy", d["end_to_end"]["exact_plate_accuracy"], p.name,
            "make plates-eval PLATES=<dir>", f"{d['name']}, n={d['end_to_end']['n']}")  # fmt: skip
    else:
        add(
            "plate_exact",
            "Exact-plate accuracy",
            None,
            None,
            "make plates-eval PLATES=<labelled Indian plate set>",
        )
    synth = [(p, d) for p, d in plates if d and d.get("synthetic")]
    if synth:
        p, d = synth[-1]
        add("plate_ocr_synthetic", "Plate OCR on synthetic crops (sanity check only)",
            d["ocr_on_crops"]["exact_plate_accuracy"], p.name, "make plates-eval", "not a real-world metric")  # fmt: skip

    ev = _load(eval_dir / "replay_events.json")
    if ev:
        add("alerts_once", "Incidents alerted more than once", float(ev["incidents_with_more_than_one_alert"]),
            "replay_events.json", "make events-replay",
            f"{ev['events']} alerts replayed on {', '.join(ev['cameras'])}; {ev['cooldown_violations']} cooldown violations")  # fmt: skip
    return out


@router.get("/metrics")
def metrics(request: Request) -> dict[str, Any]:
    eval_dir = request.app.state.config.settings.paths.data_dir / "eval"
    reports = (
        {p.name: _load(p) for p in sorted(eval_dir.glob("*.json"))} if eval_dir.is_dir() else {}
    )
    return {"headline": headline(eval_dir), "reports": reports}
