# Dwarpal progress log

## Environment

- Target: Windows 11 + WSL2 (Ubuntu), Docker Desktop (WSL2 backend), NVIDIA GPU under 8 GB VRAM.
- Python 3.11 managed by `uv`. All Python commands run inside WSL2.

## Decisions log

| # | Phase | Decision | Why |
|---|---|---|---|
| D1 | P0 | `device: auto` resolves CUDA, then MPS, then CPU. Torch is imported lazily, so the base install does not need it | ML deps are heavy; P0 has none |
| D2 | P0 | Dataset cameras default to `run_mode: cached`, webcam to `realtime`; FP16 on CUDA | Under 8 GB VRAM can't run 5 realtime streams with Re-ID + face + CLIP |
| D3 | P0 | Webcam reaches WSL2 as an RTSP stream: ffmpeg on the **Windows** side publishes the webcam (DirectShow) to MediaMTX at `rtsp://localhost:8554/webcam`. A `webcam` source type (OpenCV device index) also exists for native Linux | WSL2 has no USB webcam access without usbipd and a custom kernel |
| D4 | P0 | MediaMTX is RTSP over TCP only, with ports mapped (no host networking) | Docker Desktop on Windows doesn't support `network_mode: host` reliably; UDP RTP ports don't map cleanly |
| D5 | P0 | Python package `app` lives at `backend/app`, built by hatchling from a root `pyproject.toml` | Scripts in `scripts/` and the backend share one env and one import path |
| D6 | P0 | Sync SQLAlchemy 2 + psycopg 3. Enums are stored as VARCHAR + CHECK (`native_enum=False`) | Simple threadpool usage from FastAPI and workers; no ALTER TYPE pain in later migrations |
| D7 | P0 | Zone polygons use **normalized** `[x, y]` coordinates in `[0, 1]` | Resolution independent across datasets and webcam |
| D8 | P0 | `/health` always returns HTTP 200 with per-component status (`ok`/`degraded`) | Liveness should not flap when only the DB is down; the body says what is broken |
| D9 | P0 | `cameras.id` is a text slug (e.g. `webcam`, `ss_cam_01`), synced from `config/cameras.yaml` | YAML is the source of truth for cameras; slugs are readable in events and URLs |
| D10 | P0 | Pinned infra images: `pgvector/pgvector:pg16`, `bluenviron/mediamtx:1.21.1` | Reproducible setup |

## P0 Setup

### Plan
1. `uv` project (Python 3.11) with ruff + pytest config; package `app` at `backend/app`.
2. `docker-compose.yml`: Postgres 16 + pgvector (healthcheck, named volume) and MediaMTX (RTSP/TCP 8554, HLS 8888, API 9997).
3. Config loader (`app/core/config.py`): pydantic models for `settings.yaml`, `cameras.yaml` and `rules.yaml`, with env overrides (`DATABASE_URL`, `DWARPAL_DEVICE`, `LLM_*`, `DWARPAL_CONFIG_DIR`) and strict validation (unknown keys rejected).
4. Device resolver (`app/core/device.py`).
5. SQLAlchemy models for the section 5 schema plus the Alembic initial migration (creates the `vector` extension and HNSW cosine indexes on embedding columns).
6. FastAPI app factory with `/health` (version, DB status, pgvector status, resolved device, camera count).
7. Makefile: `setup`, `dev`, `infra-up/down`, `migrate`, `test`, `lint`, `fmt`. `streams`, `index`, `demo` and `eval` print which phase delivers them.
8. Tests: config validation, device resolution, `/health` with DB up and down, migration upgrade/downgrade round trip and a models-vs-migration drift check.

### Status
**Done.** DoD met: `make setup && make dev` works from a clean state (volume wiped,
`.venv` removed), and `pytest` is green.

Verified in the dev sandbox (Linux, Docker 29, CPU only):
- `make setup`: deps installed, containers healthy, migration `0001` applied.
- `make dev`, then `curl localhost:8000/health` returns
  `{"status":"ok","version":"0.1.0","database":{"status":"ok","pgvector":"0.8.7","revision":"0001"},"device":"cpu","cameras":1}`.
- `pytest`: **29 passed** with Postgres up; with Postgres unreachable, 23 passed and 6 DB tests skipped.
- `alembic check`: no drift between ORM models and migrations (also asserted in the test suite).
- MediaMTX: published a synthetic H.264 stream over RTSP/TCP and read it back with ffprobe
  (`h264,640,360`); it appears as ready in the `/v3/paths/list` API.
- `ruff check` and `ruff format --check` are clean.

Not verifiable in the sandbox (no GPU, not Windows): CUDA resolution under WSL2 and Docker
Desktop port mapping. Torch is not installed until P2, so `device` reports `cpu` until then.

### Schema refinements vs CLAUDE.md section 5
- `person_embeddings.quality`: crop quality, used for weighting evidence.
- `tracks.local_track_id`: per-camera tracker ID, for debugging and the GT join.
- `plate_reads.raw_text`: OCR output before normalization, for plate-accuracy analysis.
- `audit_log.details` (jsonb), e.g. the reason for an unblur.
- HNSW cosine indexes on `person_embeddings.vector`, `track_embeddings.clip` and `track_embeddings.body`.
- `severity` values: `low | medium | high | critical`.

### Next: P1 Data
Needs an HF token with the SmartSpaces terms accepted (`HF_TOKEN` in `.env`). Every download is
listed with sizes first; anything over 20 GB total needs approval.
