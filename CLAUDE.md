# Dwarpal: AI Security Operator for Gated Communities

You are the lead engineer on this repo. Read this whole file before doing anything. It is the single source of truth for scope, stack, data, phases and working rules.

## 1. What we are building

A multi-camera CCTV intelligence system for a gated society or office campus that:

1. Detects and tracks every person across multiple cameras and gives each a global ID.
2. Labels each person as **resident**, **staff**, or **unknown** (no names shown on the live view; role only).
3. Reads vehicle number plates (Indian format) and marks each vehicle **registered**, **likely registered (verify)**, or **unregistered**.
4. Raises real-time alerts: unknown person in a restricted zone, unregistered vehicle, loitering, after-hours entry.
5. Lets an operator search footage in plain English: "guy in a green t-shirt, around six foot, near the gate after 9 pm", "all unknown people today", "when did TN09AB1234 enter?"
6. Respects privacy by design: consent-based enrollment, unknown faces blurred by default, audited unblur, auto-deletion of unknown-person data.

### Demo target (what "done" looks like)
- Dashboard shows a live grid of 5 cameras: 4 replayed dataset cameras served as RTSP, plus the laptop webcam live.
- Boxes are colored by role (green resident, blue staff, red unknown, grey pending) with consistent global IDs across cameras.
- Event feed updates in real time over WebSocket.
- Search bar returns thumbnails + clips with camera and timestamp.
- Live moment: a person steps in front of the webcam and is labeled unknown with an alert. They get enrolled in about 10 seconds from 3 to 5 webcam shots, step back in, and are labeled resident.
- Metrics page shows real computed numbers (IDF1, role-label accuracy, unknown-alert precision/recall, plate accuracy). Never fabricate metrics.

## 2. Datasets (chosen, do not substitute without asking)

| Purpose | Dataset | Why |
|---|---|---|
| Primary people footage + evaluation | **NVIDIA PhysicalAI-SmartSpaces** (Hugging Face: `nvidia/PhysicalAI-SmartSpaces`, CC-BY-4.0) | Synthetic (zero real people), multi-camera, ground-truth global identities across cameras, camera calibration (enables height estimation) |
| Outdoor gate/parking realism | **MEVA** (`aws s3 ls --no-sign-request s3://mevadata-public-01/`) | Real access-controlled venue, outdoor cameras, people + vehicles, site map + camera models |
| Plate detector + OCR training/eval | **Indian-LPR** (arXiv 2111.06054, ~16k images, 4-point plate + character annotations) | Indian plate formats; fallback: an Indian plate dataset from Roboflow Universe or Kaggle |
| Vehicle gate camera stream | User's own Indian road CCTV clips, placed in `data/raw/gate_vehicles/` | Real Indian plates in motion. Ask me to drop them in when you reach Phase 5 |

Download rules:
- Never download a full dataset. Always list files first, print estimated sizes, and ask before anything over 20 GB total.
- SmartSpaces: use `huggingface_hub.list_repo_files` to inspect structure, read its README, then download **one scene with 4 to 8 cameras and many people** (prefer retail/hospital/office-like over warehouse). It may require accepting terms and an HF token.
- MEVA: pick **one date-hour slot and 2 to 3 outdoor cameras** that look like gate/parking views using the site map.
- Write an adapter per dataset that converts to a common internal format (section 6). Inspect actual annotation formats before writing parsers; do not assume.

Important reality check: faces in SmartSpaces/MEVA are tiny at CCTV distance. On datasets, identity comes mainly from **body Re-ID**; on the webcam, **face** is primary. Design the identity module to fuse both.

## 3. Tech stack

- Python 3.11, `uv` for env/deps, `ruff`, `pytest`, type hints everywhere
- Detection: latest Ultralytics YOLO (person, car, motorcycle, bus, truck); YOLO-pose optional for better upper/lower body split
- Tracking: ByteTrack or BoT-SORT (via Ultralytics or `boxmot`)
- Body Re-ID: OSNet (`torchreid` or `boxmot` weights), 512-d
- Face: InsightFace (`buffalo_l`, ArcFace 512-d). Note in README: InsightFace pretrained weights are non-commercial research use
- Plates: YOLO plate detector fine-tuned on Indian-LPR + OCR (PaddleOCR or `fast-plate-ocr`, fine-tuned if needed)
- Attributes/search: OpenCLIP (ViT-B/32 or SigLIP) image embeddings of person crops
- LLM query parsing: provider-agnostic client, model from env `LLM_PROVIDER` / `LLM_MODEL` / API key. Must have a rule-based fallback parser when no key is set
- DB: PostgreSQL 16 + pgvector, SQLAlchemy 2 + Alembic
- Backend: FastAPI, WebSocket for events, MJPEG endpoints for annotated camera streams
- Camera simulation: MediaMTX + ffmpeg looping dataset videos as RTSP streams
- Frontend: Next.js (App Router) + TypeScript + Tailwind + shadcn/ui, dark security-console aesthetic
- Infra: `docker-compose.yml` for Postgres+pgvector and MediaMTX; `Makefile` with `make setup`, `make streams`, `make index`, `make dev`, `make demo`, `make eval`
- Heavy offline work (indexing big videos, plate training) must also run as Google Colab notebooks in `notebooks/`

Device handling: auto-select CUDA, then MPS, then CPU via config. At the very start, ask me once for my OS and GPU, then set defaults.

## 4. Architecture

```
Sources (file | rtsp | webcam)
   └─> Worker per camera: detect -> track -> crops
          ├─> Person branch: body Re-ID + face (if visible) -> Global tracker (cross-camera) -> Identity engine -> Attributes (colors, height, CLIP)
          └─> Vehicle branch: plate detect -> OCR (multi-frame vote) -> normalize -> registry match
   └─> Rules engine -> Events (DB + WebSocket, with dedup/cooldown)
   └─> Indexer -> Postgres/pgvector (tracks, embeddings, plate reads, events)
FastAPI: MJPEG streams, REST, WebSocket, search
Next.js dashboard
```

Two run modes, same code path:
- **Realtime**: inference on every stream as it plays.
- **Cached**: for heavy dataset cameras, run `make index` once (locally or Colab), store per-frame results, then the worker replays video at native fps and attaches cached detections. The dashboard must look identical. The webcam always runs realtime.

### Identity engine (core logic)
- Gallery = enrolled people, each with role, consent flag, face embeddings and body embeddings.
- Each global track accumulates evidence over time: face matches (cosine above `T_face`) and body matches (cosine above `T_body`), weighted by crop quality.
- State machine per global track: `pending` -> `resident` / `staff` (match found) or `unknown` (no match after N quality observations). This prevents flicker and false unknown alerts.
- Optional staff fallback: uniform/hi-vis vest classifier, toggled in config.
- All thresholds live in `config/settings.yaml`.

### Simulated enrollment for datasets
`scripts/simulate_enrollment.py --residents 15 --staff 5 --seed 42`:
- Picks ground-truth IDs and assigns roles; everyone else is unknown.
- Extracts the best crops from an early "enrollment window" of each chosen ID and stores them in the gallery.
- Evaluation runs only on frames after the enrollment window, so there is no leakage.

### Attributes
- Upper/lower clothing color: split the person crop (or use pose keypoints), dominant color via k-means in HSV, mapped to ~11 named colors, aggregated over the best frames of the track.
- Height: use camera calibration (bbox foot point to ground plane, head point via projection) and take the median over the track. Store `height_cm` with an uncertainty band. Cameras without calibration store null; leave a hook for a 4-point manual calibration tool later.
- CLIP embedding: mean of top-k quality crops per track.

### ANPR
- Plate detect -> OCR -> normalize to Indian formats: standard `^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$` and BH series `^\d{2}BH\d{4}[A-Z]{1,2}$`.
- Vote across frames of the same vehicle track.
- Registry match: exact gives `registered`; edit distance 1 gives `likely_registered`; otherwise `unregistered` plus an event.

### Natural-language search
1. LLM (or fallback rules) parses the query into a strict JSON filter: `{entity, roles[], upper_color, lower_color, height_cm{min,max}, cameras[], zones[], time_range{from,to}, plate, free_text}`. "Six foot" maps to about 175 to 190 cm.
2. SQL filters on structured fields + pgvector cosine search using the CLIP text embedding of `free_text`.
3. Return ranked tracks with thumbnail, camera, time window, role, and a clip URL.
4. Keep 15+ test queries with expected parsed filters in `tests/search_queries.yaml`.

### Rules (YAML in `config/rules.yaml`)
- `unknown_in_zone`, `unregistered_vehicle`, `loitering` (unknown in zone longer than N seconds), `after_hours` (non-staff in restricted zone in a time window). Stretch: `tailgating`.
- Zones are polygons per camera in `config/cameras.yaml`.
- Dedup and cooldown per global ID per rule.

### Privacy
- Enrollment requires a consent flag.
- Faces of `unknown` tracks are blurred in all streams and thumbnails by default; admin unblur is written to `audit_log`.
- Retention job deletes embeddings and thumbnails of unknown tracks after N days (default 7).
- Everything runs locally; no frames leave the machine except text queries to the LLM.

## 5. Database (initial schema, refine as needed)

- `cameras(id, name, source_type, source_uri, zones jsonb, calibration jsonb)`
- `people(id, role, display_name, unit, consent bool, created_at)`
- `person_embeddings(id, person_id, kind face|body, vector vector(512), created_at)`
- `vehicles(id, plate, owner_person_id, vehicle_type, created_at)`
- `global_identities(id, role_state, person_id null, first_seen, last_seen)`
- `tracks(id, global_id, camera_id, start_ts, end_ts, upper_color, lower_color, height_cm, height_err_cm, thumb_path, quality)`
- `track_embeddings(track_id, clip vector(512), body vector(512))`
- `plate_reads(id, camera_id, ts, plate_text, confidence, vehicle_id null, status, thumb_path)`
- `events(id, rule, severity, camera_id, ts, global_id null, plate_read_id null, payload jsonb, acknowledged bool)`
- `audit_log(id, actor, action, target, ts)`

## 6. Repo structure

```
dwarpal/
  CLAUDE.md
  README.md
  Makefile
  docker-compose.yml
  config/        cameras.yaml  rules.yaml  settings.yaml
  data/          (gitignored) raw/ processed/ cache/ thumbs/ clips/
  scripts/       inspect_dataset.py  download_smartspaces.py  download_meva.sh  download_lpr.py
                 adapt_smartspaces.py  adapt_meva.py  simulate_enrollment.py  stream_cameras.sh  evaluate.py
  notebooks/     colab_index.ipynb  train_plate.ipynb
  backend/app/
    main.py  api/  core/  db/
    pipeline/    sources.py detector.py tracker.py reid.py global_tracker.py face.py
                 identity.py attributes.py height.py worker.py indexer.py
    anpr/        plate_detector.py ocr.py normalize.py registry.py
    search/      parser.py retrieval.py
    rules.py  events.py  privacy.py
  backend/tests/
  frontend/      Next.js app: /live /search /registry /events /metrics
docs/            progress.md  demo_script.md
```

Common internal data format after adapters: `data/processed/<dataset>/<camera>.mp4`, `gt/<camera>.json` (frame, global_id, bbox), `calibration/<camera>.json`.

## 7. Phases and Definition of Done

Work strictly phase by phase. At the end of each phase: run tests, update `docs/progress.md`, commit with a conventional commit message, show me exactly how to verify, and **stop and wait for "next"**.

- **P0 Setup**: uv project, compose (Postgres+pgvector, MediaMTX), Alembic migrations, config loader, `/health`. DoD: `make setup && make dev` works, `pytest` green.
- **P1 Data**: inspect + download subsets, adapters to common format, `stream_cameras.sh` serving looped RTSP. DoD: `inspect_dataset.py` prints cameras, durations, identity counts; a 10 s GT-overlay video renders correctly; RTSP streams play in VLC.
- **P2 Single-camera pipeline**: sources, detection, tracking, worker, MJPEG stream with boxes. DoD: one camera visible in browser with stable track IDs; report fps.
- **P3 Cross-camera global IDs**: OSNet + spatio-temporal constraints. DoD: IDF1 vs ground truth printed by `make eval`.
- **P4 Enrollment + identity**: gallery, identity state machine, simulated enrollment, webcam enrollment endpoint. DoD: role-label accuracy and unknown-alert precision/recall on held-out window; webcam enroll flips unknown to resident live.
- **P5 ANPR**: train or reuse plate detector/OCR (Colab notebook), normalization, registry, vehicle events. DoD: exact-plate accuracy on Indian-LPR test split; gate stream produces registered/unregistered events.
- **P6 Attributes + indexing**: colors, height, CLIP embeddings into pgvector. DoD: manual spot-check of 20 tracks for color correctness, results logged.
- **P7 NL search**: parser (LLM + fallback), retrieval, `/search` API. DoD: all queries in `tests/search_queries.yaml` parse as expected; top-5 results sensible on spot-check.
- **P8 Rules + events**: rules engine, dedup, WebSocket. DoD: loitering and unknown alerts fire once per incident.
- **P9 Frontend**: all five pages against the live backend. DoD: full demo flow works in browser.
- **P10 Privacy**: blur, audited unblur, retention job, consent enforcement. DoD: tests cover each.
- **P11 Demo polish**: `make demo` starts everything; metrics page; `docs/demo_script.md`; README with architecture diagram and GIFs.

## 8. Working rules

- Before each phase, write a short plan for that phase into `docs/progress.md`, then implement.
- Prefer simple, working, measurable over clever. Keep every model behind a small interface so it can be swapped.
- All thresholds and paths in config, never hardcoded.
- When unsure about a library API, dataset format, or version, check the installed package or official docs instead of guessing.
- Compute every metric from real runs. If something cannot be measured yet, say so.
- Keep the dashboard smooth: inference and streaming must not block each other (separate processes or threads with queues).
- Do not touch real people's data beyond the datasets listed and the user's webcam/own clips.
- If a decision is not covered here and changes scope, ask me first. Otherwise decide, note it in `docs/progress.md`, and keep moving.

Start now: ask me for OS and GPU, then begin P0.

## 9. Recorded answers (do not ask again)

- Target machine: **Windows 11 + WSL2** (Ubuntu), Docker Desktop with the WSL2 backend.
- GPU: **NVIDIA, under 8 GB VRAM**. Defaults: `device: auto` (resolves to CUDA), FP16, small YOLO weights, dataset cameras in **cached** mode, webcam in **realtime**.
- See `docs/progress.md` for every decision taken since.
