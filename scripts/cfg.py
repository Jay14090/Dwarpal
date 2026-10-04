"""Print a config value for shell scripts: `python scripts/cfg.py settings.datasets.meva.cameras`.

Lists print space-separated, scalars as-is, mappings as JSON. Paths print absolute.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.core.config import get_config


def lookup(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else getattr(obj, part)
    return obj


def render(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, list | tuple):
        return " ".join(render(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value)
    if isinstance(value, Path):
        return str(value)
    return "" if value is None else str(value)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    print(render(lookup(get_config(), sys.argv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
