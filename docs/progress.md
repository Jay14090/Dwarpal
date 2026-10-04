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

## P2 Single-camera pipeline

### Plan
- `pipeline/sources.py`: `VideoFileSource` (looped, paced at native fps, frame index = file
  position, optional shared epoch for synchronized timestamps), `CaptureSource` (RTSP/webcam with
  reconnect), `LatestFrameSource` (background reader keeping only the newest frame, so realtime
  inference skips frames instead of lagging).
- `pipeline/detector.py`: `Detector` protocol plus `YoloDetector` (Ultralytics **YOLO26**, the newest
  family in Ultralytics 8.4; person, bicycle, car, motorcycle, bus, truck; FP16 on CUDA via `quantize=16`).
- `pipeline/tracker.py`: `ByteTracker` wrapping Ultralytics ByteTrack, with separate instances for
  people and vehicles so a person is never associated with a car box.
- `pipeline/worker.py`: `CameraWorker`. One code path for realtime (detect → track) and cached
  (lookup in `data/cache/<cam>/tracks.npz`), then hooks (P3+), annotation and JPEG publishing at
  ≤ `mjpeg_max_fps`.
- `pipeline/engine.py`: all workers run in **one spawned engine process**. A `BatchingDetector`
  groups concurrent frames from realtime cameras into one forward pass. Frames reach the API over
  an `mp.Queue` (dropped when full, never blocking inference) into a `FrameHub`.
- API: `/cameras`, `/cameras/{id}/stream.mjpg`, `/snapshot.jpg`, `/frame.json`, `/pipeline/stats`,
  and a debug grid page at `/live` (the Next.js dashboard comes in P9).
- `scripts/index_cameras.py` (`make index`) builds the caches; `notebooks/colab_index.ipynb` does the
  same on a Colab GPU.

### Decisions
| # | Decision | Why |
|---|---|---|
| D17 | Detector default `yolo26s.pt` (10 M params). Sandbox and CPU-only runs use `yolo26n.pt` | s is the accuracy/speed sweet spot for an under-8 GB GPU; n is the CPU fallback |
| D18 | Engine runs in a separate process (spawn), API side only stores the latest JPEG per camera | Inference never blocks HTTP/MJPEG, and CUDA stays out of the API process |
| D19 | Realtime workers pull the newest frame at ≤ `realtime_max_fps` (15) and publish every processed frame | No stale boxes and bounded GPU load per camera |
| D20 | MJPEG box/label colors by role (`pipeline.annotate.colors`); labels show the role initial + global id (P3+), never names | CLAUDE.md: role only on the live view |
| D21 | Ultralytics usage analytics are disabled in code (`pipeline/ultra.py`, `settings.sync=false`) | Privacy: nothing leaves the machine. The sandbox proxy caught YOLO calling google-analytics.com |
| D22 | MEVA G339 stays disabled: it is a patrol PTZ (its view changes between samples) | Static zones and cached boxes don't fit a moving camera |

### Status
**Done.** DoD: cameras are visible in the browser at `http://localhost:8000/live`
(MJPEG `/cameras/<id>/stream.mjpg`) with stable track IDs. Measured in the sandbox (CPU only, no GPU):

| What | Number |
|---|---|
| Detect + track, single frame, `yolo26n` @640, CPU | 13.2 fps |
| Detect + track, single frame, `yolo26s` @640, CPU | 8.1 fps |
| `make index` (batch 4, `yolo26n`, CPU), 9000-frame MEVA clips | 19.2 fps (G336), 18.9 fps (G328) |
| Engine, realtime file camera (G339), `yolo26n`, CPU shared with an indexer | 7.3 fps processed, 128 ms/frame, 0 dropped |
| Engine, cached cameras (G336 + G328) | 30.0 fps replay each, **15.0 fps** published (cap), MJPEG ~14.5 frames/s per client |

Track-ID stability from the 5-minute caches: in G328, 6 of 7 parked cars keep **one ID for the whole
300 s** (11 car tracks are ≥ 10 s). Person tracks are short in MEVA (median 0.8-1.3 s): people are
~40 px tall at 720p and `yolo26n` on CPU misses them intermittently. The configured `yolo26s` on the
user's GPU should do better; this is re-measured with GT on SmartSpaces in P3 (per-camera IDF1).
GPU fps on the target machine is not measured yet: `make index` prints it.

Fixed during verification: CaptureSource spun in a tight reconnect loop when the webcam stream was
absent (now backs off and warns once per outage); the publish throttle delivered ~12 fps instead of 15.

The engine also wires in the P3 Re-ID/global-ID hooks (`pipeline/crosscam.py`,
`pipeline/global_tracker.py`). Without Re-ID weights they log one error and the camera runs without
global IDs; P3 below covers them.

## P3 Cross-camera global IDs

### Plan
- `pipeline/reid.py`: `ReidEncoder` interface. `OsnetEncoder` uses the **vendored torchreid OSNet**
  architecture (`app/vendor/osnet.py`, MIT) and loads official checkpoints (`osnet_x1_0_msmt17.pt`,
  downloaded via gdown from the torchreid model zoo). `ColorHistEncoder` is a weak fallback for tests.
  `crop_quality()` scores size, confidence, border truncation, aspect and occlusion.
- `pipeline/crosscam.py`: `CrossCameraHook`, per camera. It samples person tracks every
  `sample_every` frames above `min_quality`, embeds them (realtime/index) or reads `reid.npz` (cached),
  and projects foot points to the ground plane with the camera calibration.
- `pipeline/global_tracker.py`: online `GlobalTracker` shared by all cameras (rules in the module docstring):
  same-camera exclusivity, `same_place_m` agreement for concurrent calibrated sightings (plus a
  position bonus), and a `max_speed_mps` travel-time limit. Appearance is mean top-3 cosine to the
  identity gallery.
- `make index` stores Re-ID samples next to the tracks, so cached replay feeds the live global tracker exactly as realtime would.
- `app/eval/metrics.py` (IDF1/IDP/IDR, Ristani et al.) and `app/eval/mtmc.py` (offline replay of caches
  through the same hook and tracker) back `make eval` (`scripts/evaluate.py`), with `--sweep` and
  `--no-calibration` ablations. Results go to `data/eval/<dataset>_mtmc.json`.

### Decisions
| # | Decision | Why |
|---|---|---|
| D23 | Vendor OSNet instead of installing `torchreid`/`boxmot` | torchreid is a stale sdist needing compilation; boxmot pins its own detector stack. One 440-line MIT file keeps checkpoint key names |
| D24 | OSNet x1.0 / MSMT17 by default (512-d, ~2.2 M params) | Best cross-domain generalisation in the torchreid zoo; small enough to share the GPU |
| D25 | Global IDs are assigned once a track has 3 quality samples (≈ 0.5 s at 30 fps), shown as `t<id>` / pending before that | Avoids committing to an identity from one blurry crop |
| D26 | Metrics report both "online" (what the operator saw, pending frames count against) and "tracklet-level" IDF1 | Honest view of the live system plus the standard MTMC number |
| D27 | `make eval` refuses datasets without exhaustive, cross-camera GT (MEVA) | No misleading numbers (CLAUDE.md: never fabricate metrics) |

### Status
**Code complete and tested; the DoD metric is held** until SmartSpaces is available. The sandbox can't
reach Hugging Face (data) or Google Drive (OSNet weights).

Verified here:
- IDF1 implementation: unit tests for perfect tracking, id switch, cross-camera split, FP/FN and the IoU threshold.
- Global tracker: 8 scenario tests (cross-camera re-identification, pending until enough samples,
  same-camera exclusivity, far-apart concurrent sightings split, same-place concurrent sightings merge
  despite weak appearance, impossible travel speed splits, stale track expiry).
- `make eval` end to end on a synthetic 2-camera dataset (person walks A → B): tracklet-level multi-camera
  IDF1 = 1.0, online < 1.0 (pending frames), and a too-strict threshold splits the identity as expected.
- OSNet loader on torchreid-style checkpoints (`state_dict` wrapper, `module.` prefix, 4101-way
  classifier), 512-d L2-normalized output, ground projection of foot points with a known camera.
- Engine: a moving person in a realtime file camera gets global id 1 through the full hook chain.

**To produce the DoD number on the target machine:**
```bash
make data-smartspaces        # ~1.2 GB, needs HF access
make index                   # tracks + OSNet samples for every cached camera (or Colab notebook)
make eval                    # prints per-camera and multi-camera IDF1, saves data/eval/smartspaces_mtmc.json
make eval EVAL_ARGS="--sweep match_threshold=0.45,0.55,0.65"   # tune, then set it in settings.yaml
```

## P4 Enrollment + identity

### Plan
- `pipeline/face.py`: InsightFace **buffalo_l** (SCRFD + ArcFace 512-d) applied to the **head region of
  person boxes ≥ 140 px tall**. Quality = det score × size ramp × frontalness (nose between the eyes).
- `pipeline/identity.py`: `Gallery` (face/body matrices per person) and `IdentityEngine` with the state
  machine `pending → resident|staff` (evidence ≥ `accept_score`) or `→ unknown` (after
  `unknown_after_observations` quality samples without a match). Evidence per sample is weight × quality
  when cosine ≥ `t_face` / `t_body`. A switch between people needs `switch_ratio` × the current evidence
  (no flicker). On gallery change every identity's recent samples are re-scored, so **enrollment flips
  unknown → resident immediately**.
- `pipeline/person_hooks.py`: `FaceHook` (realtime, or cached `faces.npz` written by `make index`),
  `IdentityHook` (role per track) and `EnrollmentCollector` (best 3-5 face/body shots of the most
  prominent person, ≥ 0.3 s apart, plus a face thumbnail).
- `db/gallery_store.py`: enroll/delete/list/load with **consent enforced at the storage layer**. Only
  consenting people reach the gallery, and every enroll/delete goes to `audit_log`.
- Engine control channel (`EngineProcess.request`): `reload_gallery`, `enroll_capture`, `embed_images`.
  API: `POST /enroll/capture`, `POST /enroll/upload`, `GET /people`, `DELETE /people/{id}`,
  `GET /people/{id}/thumb.jpg` (actor from the `X-Actor` header for the audit log).
- `scripts/simulate_enrollment.py` (`make enroll-sim`): GT people clearly visible in the first 25 % of
  the dataset → 15 residents + 5 staff (seed 42), best 8 crops each, stored in Postgres
  (`unit = SIM:<dataset>`, replaced each run) and `processed/<dataset>/enrollment/gallery.npz`.
- `app/eval/identity_eval.py` via `make eval`: **role-label accuracy** (decided observations, coverage and
  confusion matrix) and **unknown-alert precision/recall** on frames after the window only (no leakage).

### Decisions
| # | Decision | Why |
|---|---|---|
| D28 | Face search only on person boxes ≥ 140 px tall, head region only | Cost scales with close people; CCTV-distance faces are too small to help (CLAUDE.md reality check) |
| D29 | Unknown alerts are counted once per global identity (first `pending → unknown`) | Matches the rules-engine dedup (P8); "once per incident" |
| D30 | Webcam enrollment captures in the engine (raw frames, real encoders) instead of from MJPEG JPEGs | MJPEG frames carry drawn boxes and lossy re-encoding |
| D31 | Uploaded-photo enrollment: best face per photo; a photo with h/w ≥ 1.6 also adds a body sample | Portrait photos have no usable body |
| D32 | No authentication in the demo; the `X-Actor` header names the actor in `audit_log` | Out of scope; P10 audits admin actions with it |

### Status
**Code complete and tested; the DoD numbers are held** until SmartSpaces is available (same blocker as P3).
The live webcam flip needs the user's webcam.

Verified here (102 tests):
- Identity state machine: unknown after N unmatched samples, face match → role, quality weighting,
  low-quality samples ignored, no flicker on one contrary sample, switch on sustained evidence,
  **enrollment flips unknown → resident immediately**, deleting a person resets their tracks.
- Engine integration with fakes: a person on a realtime camera becomes **unknown**, a 1.5 s capture
  collects 3-5 face shots and ≥ 3 body shots plus a JPEG thumbnail, and after the gallery reload the same
  live track turns **resident**.
- API on a real migrated Postgres: consent refused before any capture, person + 8 embeddings + thumbnail
  + audit row stored, too-few-shots hint, photo upload, audited delete, 503 without the engine, and
  non-consenting rows never loaded into the gallery.
- `simulate_enrollment.py` end to end on a synthetic 2-camera dataset with real video files.
- `evaluate_identity` on the synthetic dataset: role accuracy 1.0 with one correct unknown alert per
  unenrolled person, and it reports enrolled people wrongly alerted as unknown.
- Live: the API → engine control channel round trip in a real engine process. The gallery loads from
  Postgres, consent is enforced, and a capture with no usable person returns the "stand closer" hint.
- InsightFace buffalo_l downloads and runs (SCRFD on a 320 px head region ~110 ms on CPU). There are no
  privacy-safe face images in the sandbox (dataset faces are tiny), so face *recognition* is verified on
  the user's webcam.

**To produce the DoD numbers on the target machine:**
```bash
make data-smartspaces && make index     # once (P3)
make enroll-sim                         # 15 residents + 5 staff, seed 42
make eval                               # P3 IDF1 + P4 role accuracy and unknown precision/recall
```
**Live webcam check:** run `scripts/webcam_publish.ps1` on Windows, then `make dev` and open
`http://localhost:8000/live`. Step in front of the camera (red **U** box after ~1 s), then:
```bash
curl -X POST localhost:8000/enroll/capture -H 'content-type: application/json' \
  -d '{"camera_id":"webcam","role":"resident","display_name":"You","consent":true}'
```
Look at the camera for ~6 s; the box turns green **R** without leaving the frame.
