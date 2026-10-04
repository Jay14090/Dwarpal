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

## P1 Data

### Sandbox network findings (affects what can be verified here)
- Reachable: PyPI, GitHub (incl. release assets), raw.githubusercontent, AWS S3 (MEVA bucket), GCS.
- Blocked: huggingface.co **and its CDN** (SmartSpaces data, OpenCLIP/HF weights), arXiv, Kaggle,
  Roboflow, Google Drive, Kitware GitLab, download.pytorch.org.
- Consequence: SmartSpaces scripts are written against the README-documented formats, and the
  real `calibration.json` / `calibration_2025_format.json` contents were read through a web
  scraper (see below). The actual download runs on the user's machine. MEVA is fully downloaded
  and adapted in the sandbox.

### Inspected formats (real files, not assumed)
- SmartSpaces `MTMC_Tracking_2024/<split>/scene_XXX/`: `camera_XXXX/{video.mp4,calibration.json}`,
  scene-level `ground_truth.txt` (`camera_id obj_id frame_id xmin ymin w h xworld yworld`, frame 0-based,
  obj_id consistent across cameras), `calibration_2025_format.json`, `ground_truth_2025_format.json`.
  Per-camera `calibration.json` keys: `"camera projection matrix"` (3x4), `"homography matrix"` (3x3,
  ground plane z=0 → image), `reprojection_error`. 2025-format calibration adds `intrinsicMatrix`,
  `extrinsicMatrix` (3x4 [R|t]), `cameraMatrix`, `homography`, fps/frameWidth/frameHeight attributes.
  Videos 1920x1080 @ 30 fps. Scene 071 (retail) has 16 cameras of 115-304 MB each; `camera_0649` is corrupt (README).
- MEVA KF1: videos `drops-123-r13/<date>/<hour>/<date>.<start>.<end>.<site>.<cam>.r13.avi`
  (H.264, 1920x1072, 30 fps, 5 min). Annotations exist only for some clips (`examples/annotations/`)
  in KPF YAML: `geom.yml` (`id1` track, `ts0` frame, `g0` x1 y1 x2 y2), `types.yml` (`id1` → Person/Vehicle),
  `activities.yml`. **They are activity-centric: only actors in an annotated activity have boxes**,
  so MEVA GT is sparse and cannot score tracking. Track ids are per clip, not cross-camera.
- MEVID (`mevid-annotations/`) is cropped-person images only (14-32 GB tarballs), no full-frame
  boxes, so it is not usable for our GT.

### Choices
| # | Decision | Why |
|---|---|---|
| D11 | MEVA slot **2018-03-11 13:50**, cameras **G328** (parking 2/3 + carport), **G336** (school exterior / roads), **G339** (parking PTZ). 465 MB | Only annotated slot with three outdoor parking/road views (site map) |
| D12 | SmartSpaces: `MTMC_Tracking_2024/test/scene_071` (retail, 16 cams). The downloader fetches `ground_truth.txt` first, ranks cameras by person count and overlap, then downloads the top 6 (default) videos plus calibration. About 1.2 GB | People-only retail scene matches "prefer retail/hospital/office"; 2024 GT is exhaustive with cross-camera ids |
| D13 | Adapters transcode to H.264, at most 720p, GOP = 1 s, no B-frames; GT boxes and calibration are rescaled to match | Cheap to decode on an under-8 GB GPU and loopable as RTSP with `-c copy` (no re-encode while streaming) |
| D14 | Common GT: `gt/<camera>.json` with `rows: [frame, global_id, x1, y1, x2, y2, class]`, plus flags `exhaustive` and `cross_camera_ids` | Evaluation can refuse to score sparse GT (MEVA) instead of producing misleading numbers |
| D15 | Common calibration: `calibration/<camera>.json` with `K`, `R`, `t`, `P` (3x4) and `H` (ground → image). MEVA has none (camera models live on Kitware GitLab, blocked); stored as null | Height estimation needs P; null means "no height" per the spec |
| D16 | Cached-mode cameras read the processed **file** directly (paced at native fps, looped) so frame indices line up with cached detections. MediaMTX RTSP serves the same files for VLC and realtime mode | An RTSP reader can't recover the source frame index after a loop |

### Status
**Done in the sandbox for MEVA. SmartSpaces download is held** (Hugging Face is blocked here). It runs on the user's machine with `make data-smartspaces`.

Verified here:
- `scripts/download_meva.sh`: lists sizes (3 clips, 0.47 GB), downloads videos, annotations and the site map. Idempotent.
- `scripts/adapt_meva.py`: 3 cameras transcoded to 720p H.264 (1 s GOP, no B-frames). GT rescaled.
- `make inspect`:
  ```
  camera              res   fps  dur(s)  frames   boxes persons vehicles calib
  meva_g328      1290x720  30.0   300.0    9000    1056       2        5 no
  meva_g336      1280x720  30.0   300.0    9000    1375       3        3 no
  meva_g339      1280x720  30.0   300.0    9000     335       2        0 no
  ```
- `scripts/render_gt_overlay.py meva meva_g336`: 10 s overlay rendered, and frames visually checked;
  boxes sit on the annotated vehicles after rescaling. A person standing next to a car in G336 has no
  box, which confirms MEVA GT is sparse (D11/D14).
- `make streams`: three RTSP loops through MediaMTX. `ffprobe rtsp://localhost:8554/meva_g328` gives
  `h264,1290,720,30/1`. A 4 s clip looped across 10 s of RTSP reading (285 frames, no errors), so the
  `-c copy` loop is seamless.
- SmartSpaces: downloader and adapter are tested against real `scene_071` calibration (fixture) with
  a mocked hub plus a generated 1080p video, covering selection (excluding corrupt cam 649), the
  >20 GB confirmation gate, box/calibration rescaling, and the projection round trip.
- On the real camera_0635 calibration, P = s·K[R|t], the camera centre is 4.0 m above the ground and
  world z points up. Height estimation (P6) can use it directly.

Not verifiable here: VLC playback (no GUI; ffprobe/ffmpeg read the same RTSP streams instead) and the
real SmartSpaces download/adaptation.

Added: `scripts/webcam_publish.ps1`, which publishes the Windows webcam (DirectShow) to
`rtsp://localhost:8554/webcam` for the WSL2 worker (D3). `scripts/register_cameras.py` appends
adapted SmartSpaces cameras to `cameras.yaml`.

Note for the user's GPU: PyPI `torch` 2.14 ships CUDA 13 wheels. These need a recent Windows NVIDIA
driver (R580+) and a Turing-or-newer GPU (GTX 16xx / RTX). On a Pascal card (GTX 10xx), install
from the cu126 index instead (see README).
