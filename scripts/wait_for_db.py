"""Block until the configured Postgres accepts connections (used by `make setup` / `make dev`)."""

from __future__ import annotations

import argparse
import sys
import time

from app.core.config import get_config
from app.db.session import check_database, make_engine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds to wait")
    args = parser.parse_args()

    engine = make_engine(get_config().settings.database)
    deadline = time.monotonic() + args.timeout
    while True:
        status = check_database(engine)
        if status["status"] == "ok":
            print(f"database ready (pgvector={status['pgvector']}, revision={status['revision']})")
            return 0
        if time.monotonic() > deadline:
            print(f"database not reachable after {args.timeout:.0f}s: {status}", file=sys.stderr)
            return 1
        time.sleep(1.0)


if __name__ == "__main__":
    sys.exit(main())
