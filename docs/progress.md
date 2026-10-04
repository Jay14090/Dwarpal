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

## P5 ANPR

### Plan
- **Reuse pretrained models** ("train or reuse"): plate detector `yolo-v9-t-640-license-plate-end2end`
  (open-image-models, ONNX) and OCR `cct-s-v2-global-model` (fast-plate-ocr, ONNX). Both download from
  GitHub release assets and sit behind `PlateDetector` / `PlateOcr` interfaces. A fine-tuned Ultralytics
  detector (`backend: yolo`) or a fine-tuned OCR ONNX (`ocr.onnx_path`) drop in through config.
- `anpr/normalize.py`: template fitting to the Indian formats (standard `^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$`,
  BH `^\d{2}BH\d{4}[A-Z]{1,2}$`). Confusable characters (O/0, I/1, B/8, Z/2, S/5, …) are corrected only
  where the template demands, at a cost of one each; the cheapest fit wins, and a valid RTO state code breaks ties.
- `anpr/voting.py`: per-vehicle-track vote weighted by OCR confidence (≥ 3 agreeing reads, ≥ 50 % share),
  settled when the track leaves; reads with any character below `min_char_conf` are dropped.
- `anpr/registry.py`: exact → `registered`, edit distance 1 → `likely_registered` (verify), else `unregistered`.
- `anpr/hook.py` (`PlateHook`) on cameras with `anpr: true`. It finds the plate in an expanded vehicle crop, then
  OCR, normalize and vote. Cached cameras replay raw reads from `plates.jsonl` written by `make index`.
- `anpr/service.py` + `pipeline/indexer.py` (`DbWriter`, background Postgres writer) + `events.py` (`EventBus`
  with per-rule cooldown from `rules.yaml`): plate reads go to `plate_reads` with a thumbnail, and an
  `unregistered_vehicle` event goes to `events` and the API (WebSocket in P8).
- API: `GET/POST/DELETE /vehicles` (plates normalized and validated, audited), `GET /plates` (+ thumbnails),
  `GET /events`. Cameras are synced from `cameras.yaml` into the DB at API and engine start.
- Scripts: `eval_plates.py` (`make plates-eval`), `download_lpr.py` (Kaggle / Roboflow), `make_plate_ocr_dataset.py`
  (synthetic Indian plates), `adapt_gate_clips.py` (`make gate`), and `notebooks/train_plate.ipynb` (OCR + detector
  fine-tuning on Colab, exported to ONNX / `.pt`).

### Decisions
| # | Decision | Why |
|---|---|---|
| D33 | **Indian-LPR is not public** (authors withhold it for legal reasons), so plate accuracy is measured on any text-labeled Indian set the user provides (`images/` + `labels.csv`) | CLAUDE.md fallback; the Kaggle `kedarsai` set (CC0) has boxes only, no plate text, so it scores the detector only |
| D34 | Reuse the pretrained global OCR first and fine-tune via Colab when labeled data exists | India is not among its 65 training regions, but the alphabet and 10 slots fit Indian plates; a synthetic sanity check reads them well (below) |
| D35 | Synthetic plates are for sanity checks and OCR pre-training only, never a reported metric; synthetic *scenes* were dropped from evaluation | Flat shapes on noise gave a meaningless detector recall (12 %) |
| D36 | Plate events dedup per plate per rule cooldown; event `global_id` lives in the payload until P6 stores global identities | `events.global_id` is an FK to `global_identities` |
| D37 | Gate clips are kept at up to 1080p (`adapt_gate_clips.py`), dataset cameras at 720p | Plates need pixels |

### Status
**Code complete and tested. The DoD numbers are held, waiting on user input:**
1. **A text-labeled Indian plate test set** for exact-plate accuracy (Indian-LPR is unavailable). Drop it in
   `data/raw/plates_eval/<name>/` (`images/` + `labels.csv`: `filename,plate[,x1,y1,x2,y2]`) and run
   `make plates-eval PLATES=data/raw/plates_eval/<name>`.
2. **Your gate CCTV clips** in `data/raw/gate_vehicles/` → `make gate` registers them as ANPR cameras; then
   `make dev` and register a few plates with `POST /vehicles` to see `registered` / `unregistered` events.

Verified here (sandbox, CPU):
- Pretrained detector + OCR download and run (detector ~300 ms per 640 px frame on CPU, OCR batched).
- **Synthetic sanity check (not the DoD metric):** OCR on 300 synthetic HSRP-style Indian plate crops: exact-plate
  **84.7 %** after normalization (84.3 % raw), character accuracy 97.6 %, valid-format rate 93.7 %.
- 15 normalization cases, Levenshtein and registry statuses, the voter (noisy reads → right plate, invalid and
  low-confidence reads ignored, a leaving track settles its vote), identical decisions from cached replay, unregistered
  event cooldown, stream labels (`MH12AB1234 UNREG`), plate read + event rows in Postgres, an ANPR camera in a live
  engine emitting plate and event messages, the vehicles/plates/events API, and the gate-clip adapter.

## P6 Attributes + indexing

### Plan
- `pipeline/attributes.py`: upper/lower clothing colour. The person box is split (torso 15–50 %, legs 55–90 %,
  central half of the width). Dominant colour comes from k-means in HSV (circular hue), mapped to 11 names, and
  each track votes over its best 5 crops weighted by crop quality.
- `pipeline/height.py`: the box foot point goes to the ground plane through the calibration, the head point
  comes from projection, and the result is the median ± MAD over the track (minimum ±2 cm, 120–220 cm). A camera
  without calibration stores null.
- `pipeline/clip.py`: OpenCLIP ViT-B/32 (LAION-2B) embeddings, the mean of the top-k quality crops per track,
  plus `embed_text` for search.
- `pipeline/indexer.py` (`TrackIndexer` hook) closes a track when the tracker drops it and writes
  `tracks` (colours, height, zones visited, role, frame range), `track_embeddings` (CLIP + body),
  `global_identities` and a thumbnail through the background `DbWriter`.
- Migration `0002`: `tracks.role`, `tracks.zones` (jsonb), `start_frame/end_frame` and HNSW cosine indexes.
- `scripts/index_tracks.py` (`make index-db`): replays cached cameras through the same hooks into Postgres.

### Decisions
| # | Decision | Why |
|---|---|---|
| D38 | CLIP weights come from Hugging Face, else the OpenCLIP GitHub release (`vit_b_32-laion2b_e16`) | HF is blocked in the sandbox; same architecture and training data |
| D39 | Added indoor MEVA cameras G421 ("Clubhouse cafe") and G299 ("Clubhouse gym"), same 13:50 slot, **disabled** by default | The outdoor cameras gave only 8 person tracks, too few for the 20-track spot-check; indoor people are 60–280 px tall |
| D40 | The lower-body colour drops pixels that match the box's side strips (background) before k-means; the upper body does not | Spot-check (below): legs are thin and the wooden gym floor made "yellow pants". Filtering the torso too removed real clothing pixels (2 regressions on the same crops) |

### Spot-check: 20 tracks for colour correctness (DoD)
20 random tracks (seed 6) from MEVA G299/G421/G336, judged by eye from each track's best crop
(contact sheet: `docs/img/p6_color_spotcheck.jpg`). These were judged against the crops, not against ground truth;
MEVA has no clothing labels.

| | before D40 | after D40 |
|---|---|---|
| Upper correct (of 19 judgeable; 1 box holds two people) | 16 (84 %) | 16 (84 %) |
| Lower correct (of 18; legs hidden behind a table on 2) | 14 (78 %) | 17 (94 %), or 16 (89 %) counting dark denim → "black" as wrong |

- Errors left: red tops in crowded boxes read as black (the dark person behind dominates; 2 cases), a
  light-blue shirt reads as green (1), and light trousers read as black (1).
- **Caveat:** this footage is mostly dark clothing, so an "always black" guess scores 53 % upper and about 90 % lower.
  The lower-body number therefore says little on this clip. SmartSpaces has more varied clothing; re-run the
  spot-check there.
- Height: MEVA cameras are uncalibrated here, so `height_cm` is null (by design). It is tested with the real
  SmartSpaces calibration fixture.

### Status
Done. 177 tracks are indexed (G336 8, G421 8, G299 161) with colours, CLIP and thumbnails; `make index-db` runs at
about 55 frames/s per camera (CPU, CLIP included). G299 fragments heavily: 161 tracks in 5 minutes of a crowded gym
with `yolo26n` on CPU.

## P7 Natural-language search

### Plan
- `search/parser.py`: a strict `SearchFilter` (pydantic, `extra=forbid`) with `{entity, roles[], upper_color,
  lower_color, height_cm{min,max}, cameras[], zones[], time_range{from,to}, plate, free_text}`.
  - **RuleParser** (always available): plate regex + Indian normalization; time phrases in `app.timezone`
    (today, yesterday, after/before/between/around X, last N hours/minutes, morning/evening/night); heights
    ("six foot" → 175–190, 5'8", 180 cm, tall/short); colour words bound to the nearest garment (a lone colour is
    upper); roles and synonyms; camera names and multi-word zone names matched first ("resident parking" is a
    place, not a role); the leftover appearance words become `free_text`.
  - **LlmParser**, provider-agnostic. `LLM_PROVIDER=anthropic` uses the official `anthropic` SDK with structured
    outputs (`messages.parse(output_format=SearchFilter)`, default model `claude-opus-5-5`, `output_config.effort`
    from settings, refusal check). `openai` speaks any OpenAI-compatible endpoint (`LLM_BASE_URL`: OpenAI, Groq,
    Ollama, …) in JSON mode. The output is validated against the same schema, and **any failure falls back to the
    rules**. Only the query text, camera/zone names and the current time are sent.
- `search/retrieval.py`: SQL filters (identity role, colours, height band ± error, cameras, zones `?|`, time
  overlap) ranked by pgvector cosine distance to the CLIP text embedding of `free_text`. If colour filters leave
  nothing, they are relaxed and the response says so. Vehicles search `plate_reads`: exact plate first, then edit
  distance 1; status from roles; a zone narrows to the cameras that contain it.
- API: `POST/GET /search`, `POST /search/parse`, `GET /tracks/{id}/thumb.jpg`, and `GET /tracks/{id}/clip.mp4`
  (an ffmpeg cut of the track window ± 1 s, cached). The CLIP text embedding comes from the engine process
  (`embed_text` command) or, if the engine is off, a lazily loaded local encoder.
- `tests/search_queries.yaml`: 18 queries with expected filters at a fixed "now".

### Decisions
| # | Decision | Why |
|---|---|---|
| D41 | The rule parser is the default; the LLM is opt-in through env (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL`) | CLAUDE.md requires working search with no key; nothing leaves the machine unless configured |
| D42 | Colour filters are hard, but they are relaxed (and flagged) when they would return nothing and there is free text | Colour estimates are wrong about 15 % of the time (P6); CLIP still ranks the right person first (spot-check query 2) |
| D43 | Role filters use the identity's *current* role (`global_identities.role_state`), falling back to the track's | An enrollment after the fact should make old footage searchable as resident |
| D44 | Clips are cut only for file cameras; the webcam keeps no recording | Privacy by design; there's nothing to cut from a live stream |

### Status
Done. All 18 queries in `tests/search_queries.yaml` parse to the expected filters (`make search-check`). The
Anthropic and OpenAI-compatible paths are tested with mocked clients (structured-output call, refusal → rules,
schema-violating JSON → rules); no real LLM call was made here (no key). Retrieval and API are tested in Postgres
with pgvector ranking, relaxation, exact and fuzzy plates, and clip cutting.

**Top-5 spot-check** on the indexed MEVA tracks (rule parser + CLIP, `docs/img/p7_search_spotcheck.jpg`),
relevant results by eye:

| Query | Relevant in top 5 |
|---|---|
| person in a blue denim jacket | 3 (blue tops; 1 actually denim) |
| man in a white shirt sitting at a table | 4 (colour relaxed; same man in 4 track fragments) |
| woman in a red top | 3 |
| person wearing a hat | 3 |
| person in a light blue shirt | 2 |
| someone in a long black coat | 3 (all 5 wear black outerwear) |

That's 18/30 (60 %) relevant. CLIP ViT-B/32 on small, low-light CCTV crops is the limit; scores sit in a narrow
0.30–0.39 band.

## P8 Rules + events

### Plan
- `app/rules.py`: `RulesEngine`, shared by all cameras, plus a per-camera `RulesHook` after `IdentityHook`.
  - A *presence* is kept per (person, zone); a person is their global id, or camera + track while unassigned.
    A presence ends after `presence_gap_s` (10 s) unseen.
  - Each rule fires **at most once per presence**, and the `EventBus` cooldown (per rule, per person, 300 s)
    blocks re-alerts on quick re-entry.
  - Rules:
    - `unknown_in_zone`
    - `loitering` (dwell ≥ `min_dwell_s`)
    - `after_hours` (local-time window that may cross midnight; non-staff by default)
    - `tailgating` (stretch: an unknown person enters within `window_s` after a resident or staff; disabled)
  - `params.roles` picks the identity states a rule applies to; `pending` never triggers by default.
- Events get a person-crop thumbnail (`thumbs/events/{id}.jpg`), a real `global_id` FK (the identity is
  upserted), and the payload `{zone, role, track_id, dwell_s}`.
- API (`api/events.py`): `GET /events` (filter by rule, `unacknowledged`), `POST /events/{id}/ack` (audited),
  `GET /events/{id}/thumb.jpg`, and `WS /ws/events`, which fans out engine events and plate reads through
  `FrameHub.listeners` with a per-client queue that detects disconnects.
- `scripts/replay_events.py` (`make events-replay`) replays cached cameras through identity + rules. Next to the
  rules engine it keeps an independent presence tally, and it checks (1) no incident gets two alerts of the same
  rule and (2) no cooldown violation. `GET /metrics` serves the saved reports only.

### Decisions
| # | Decision | Why |
|---|---|---|
| D45 | One alert per incident (continuous presence), on top of the per-person cooldown | A cooldown alone re-fires every 5 minutes on someone who stays 20 minutes |
| D46 | `pending` people never trigger rules unless a rule lists `pending` in `params.roles` | The identity state machine exists to avoid false unknown alerts. Note: on far cameras (MEVA outdoor people are 30–47 px, below the 64 px Re-ID quality floor) nobody resolves, so those cameras alert only if you opt in |
| D47 | The MEVA G328 `reid.npz` was an empty leftover of an aborted run (older than `tracks.npz`) and was deleted | It hid every person from the global tracker |

### Status
DoD met on real footage. `make events-replay CAMERAS="meva_g421 meva_g299" REPLAY_ARGS="--reid-backend colorhist --no-db"`:
- **Daytime** (clip clock): 6 people resolved to unknown. That gave 5 `unknown_in_restricted_zone` alerts (gym;
  the sixth person was only in the unrestricted cafe) and 1 `loitering_unknown` at exactly 60 s dwell.
  **0 incidents with more than one alert, 0 cooldown violations** (`data/eval/replay_events.json`).
- **Night** (`--start 2026-10-04T22:30:00+05:30`): 11 alerts, made of 5 `after_hours_entry` + 5 unknown-in-zone +
  1 loitering. Again one per person per rule, with 0 duplicates and 0 cooldown violations
  (`data/eval/replay_events_night.json`).
- **Caveat:** the colour-histogram Re-ID fallback (OSNet weights can't download here) merges about 160 gym
  tracks into 6 global IDs, because most people wear dark clothes. That shows the once-per-incident property, but
  **alert recall is not meaningful here**. Re-run with OSNet on your machine.
- 11 rules-engine tests: once per incident over 200 s, detection gaps, cooldown on re-entry, dwell reset,
  after-hours window across midnight, tailgating timing, role gating. Events API tests cover the global-id link,
  thumbnail, idempotent audited ack, and WebSocket fan-out and cleanup.

## P9 Frontend

### Plan
- `frontend/`: Next.js 16 (App Router), TypeScript, Tailwind 4, and a dark security-console theme with shadcn/ui
  token names. The five pages are client components talking to the FastAPI backend (CORS already allows
  `localhost:3000`):
  - `/live`: MJPEG tiles, per-camera role counts from `frame.json`, the WebSocket event feed, and webcam enrollment (consent checkbox required).
  - `/search`: query box + examples, parsed filter chips, "relaxed" notice, thumbnail grid, clip player, plate rows.
  - `/events`: history, rule filter, open-only, ack, live plate reads.
  - `/registry`: people (thumbnail, samples, consent, delete), photo enrollment, vehicles CRUD with owners.
  - `/metrics`: headline cards from `GET /metrics`, raw reports.
- `lib/use-live-feed.ts`: WebSocket with exponential-backoff reconnect.

### Decisions
| # | Decision | Why |
|---|---|---|
| D48 | shadcn/ui-style components are written by hand (`cva` + `tailwind-merge`, same API and tokens), and `components.json` is included | The shadcn registry (ui.shadcn.com) is blocked in the sandbox; `npx shadcn add …` works on your machine and fits the theme |
| D49 | System font stacks, no `next/font/google` | Builds offline; nothing is fetched from Google ("everything runs locally") |
| D50 | The browser talks to the API directly (`NEXT_PUBLIC_API_URL`) rather than through Next rewrites | MJPEG and WebSocket stream straight from FastAPI; no proxy buffering |

### Status
Done for everything that can run here. `make web-build` passes (tsc, eslint, `next build`, all routes). I drove
every page with headless Chromium against the real backend, running 4 cached MEVA cameras with the rules engine on:
- `/live`: 4 live tiles with boxes and role counts.
- `/search`: "man in a white shirt sitting at a table" → parsed by rules, colour relaxed, CLIP-ranked; the same
  seated man takes the top 4.
- `/events`: thumbnails; Ack dimmed the row and wrote `audit_log`. A new `unknown_in_restricted_zone` alert
  arrived over the WebSocket while the page was open.
- `/registry`: adding "tn 09 ab 1234" stored `TN09AB1234`.
- `/metrics`: real numbers or "not measured" with the command.

No console errors, and no horizontal scroll at 390 px width. Screenshots: `docs/img/ui_*.jpg`.

**Held for you:**
- The webcam enrollment flip (unknown → resident) needs your webcam (`scripts/webcam_publish.ps1`).
- Clip playback was checked with curl + ffprobe (H.264 720p), not in the browser: Playwright's Chromium has no
  H.264 decoder, so check it in your browser.

## P10 Privacy

### Plan
- `app/privacy.py`:
  - `head_box`: the top 24 % of the person box, padded.
  - `blur_heads` / `blur_crop`: pixelate, then Gaussian blur; irreversible on the output.
  - `PrivacyPolicy`: which roles are blurred, who is an admin.
  - `run_retention` and `retention_loop`.
- **Streams:** `CameraWorker._publish` blurs heads of people whose role is in `privacy.blur_roles` (default
  `unknown`, `pending`) before drawing boxes. An admin can unblur one camera for at most `unblur_max_s`
  (`POST /privacy/unblur-stream` with a reason → engine `unblur` command, audited).
- **Thumbnails** (tracks, events): stored once, blurred **when served** from the identity's current role, so
  enrolling someone later unblurs their history automatically. `?unblur=true` is admin-only (403 otherwise) and
  audited (`unblur_thumbnail`).
- **Clips** (`app/clips.py`): decoded frame by frame, heads of every person blurred using the camera's cached
  detections (the track's own person stays visible only if identified), then piped to ffmpeg H.264. With no
  detections the clip is refused (409) rather than served unanonymised. Admin `?unblur=true` is audited and the
  file is deleted after sending.
- **Retention:** an API background thread (every `retention_interval_s`) plus `POST /privacy/retention` (admin)
  delete embeddings, thumbnails and clips of unknown/pending tracks, and the person crops of their events, older
  than `unknown_retention_days` (7). Track rows (camera, time, colours) and alert records stay. Each run writes an
  audit row with counts.
- **Consent:** enrollment without consent → 422 on both capture and upload; the gallery loads only consenting
  people (P4 tests).
- `GET /audit` (admin). UI: an operator/admin switch in the nav, an eye button on search cards and live tiles
  (admin), and a privacy & audit card with "Run retention now" on `/registry`.

### Decisions
| # | Decision | Why |
|---|---|---|
| D51 | Blur `unknown` **and** `pending` | Pending people haven't been identified as consented residents or staff; blurring only "unknown" would show faces until the state machine decides |
| D52 | Blur a head region derived from the person box, not detected faces | Faces are 5–20 px at CCTV distance and the face detector misses them; the head region always exists |
| D53 | Thumbnails are blurred at serve time instead of storing a blurred and a raw copy | One file per thumbnail; the current identity decides; admin unblur needs no second store |
| D54 | Admin = `X-Actor` in `privacy.admin_actors` | There's no authentication in scope (demo); the header model keeps every action attributable in `audit_log`. Real auth should replace it before any deployment |

### Status
Done. The tests cover each DoD item (`test_privacy.py`, `test_events_api.py`, `test_people_api.py`):
- head blur only touches the head;
- the stream blurs unknown and pending but not residents; the unblur window works and expires;
- clips are blurred, with the identified person kept;
- thumbnails are blurred by default; admin unblur is audited and non-admins get 403; enrollment unblurs history;
- retention deletes only old unknown/pending data, and is audited and admin-gated;
- the stream-unblur endpoint requires admin;
- consent is refused on capture and upload, and the gallery loads consenting people only.

In the browser (MEVA gym and cafe): stream heads blurred (`docs/img/p10_stream_blur.jpg`), search thumbnails
blurred, the admin eye unblurs one card, and the audit log shows the `unblur_thumbnail` row
(`docs/img/p10_search_blur.jpg`).

## P11 Demo polish

### Plan
- `make demo` (`scripts/demo.sh`) brings up infra, migrations, a cache check per enabled camera, a dashboard
  install/build when needed, then the API + engine and the dashboard together. It prints the URLs, and Ctrl-C
  stops everything.
- The `/metrics` page already shipped in P9; it reads only `data/eval/*.json`.
- `docs/demo_script.md`: an 8-minute walkthrough (live view → webcam unknown → enroll → resident → search →
  alerts, ack and audited unblur → metrics) plus troubleshooting.
- README: Mermaid architecture diagram, two GIFs recorded from the running system (`docs/img/live.gif`,
  `docs/img/search.gif`), quickstart, a feature table, measured numbers, a "waiting on you" table and licences.

### Decisions
| # | Decision | Why |
|---|---|---|
| D55 | The indoor MEVA cameras G421 (cafe) and G299 (gym, restricted) are now **enabled** by default (revises D39) | The demo target is 4 replayed cameras + the webcam. Outdoor MEVA people are too small to identify, so the indoor cameras carry the identity and alert story until SmartSpaces is downloaded |
| D56 | `make demo` serves the production dashboard build (`next start`) and rebuilds only when `frontend/src` changed | Smooth UI during the demo; the dev server recompiles on first visit to each page |

### Status
Done. `make demo` was run here: infra, migrations, the cache check (all enabled cameras ok), API (health ok) and
dashboard (`/live` 200) came up, and Ctrl-C left no processes behind. The GIFs and screenshots come from the
running system. Backend: 202 tests pass, and ruff is clean. Frontend: `make web-build` passes.

## Where things stand (all phases)
| Phase | State |
|---|---|
| P0 Setup | Done |
| P1 Data | Done for MEVA (5 cameras). The SmartSpaces download is scripted and tested, but **held**: Hugging Face is blocked here |
| P2 Single-camera pipeline | Done (CPU: 13 fps `yolo26n`, 8 fps `yolo26s`; GPU numbers on your machine) |
| P3 Cross-camera IDs | Code done. **IDF1 held**: it needs SmartSpaces (MEVA ground truth is too sparse) and OSNet weights |
| P4 Enrollment + identity | Code done. **Role accuracy and unknown-alert P/R held** (SmartSpaces); **webcam flip held** (your webcam) |
| P5 ANPR | Code done. **Exact-plate accuracy held** (a text-labelled Indian set); **gate stream held** (your clips) |
| P6 Attributes + indexing | Done; 20-track spot-check logged |
| P7 NL search | Done; 18/18 parser queries, top-5 spot-check logged |
| P8 Rules + events | Done; once per incident verified on real footage |
| P9 Frontend | Done; all five pages verified in a browser |
| P10 Privacy | Done; tests cover blur, unblur audit, retention and consent |
| P11 Demo polish | Done |
