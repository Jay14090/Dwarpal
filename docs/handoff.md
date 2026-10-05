# Handoff: continuing Dwarpal on your machine

This file is for a **new Claude Code session** (or you) picking the project up with no prior conversation.
Read `CLAUDE.md` (spec) and then this file; `docs/progress.md` has every decision (D1–D56) and measured number.

## State in one paragraph
All phases P0–P11 were built in a cloud sandbox (CPU only, no Hugging Face / Google Drive / webcam access) and
pushed to branch **`claude/sleepy-ptolemy-2xcbeh`** of `github.com/Jay14090/Dwarpal` (not merged into `main`).
202 backend tests pass, ruff is clean, the Next.js dashboard builds, and every page was driven in a real
browser. What is **not done** is everything that needs your machine: GPU runs, the SmartSpaces dataset, OSNet
weights, the webcam, your gate clips, a labelled Indian plate set. Those produce the headline metrics, which
are therefore still "not measured". The work left is mostly **running things and fixing what breaks**, not
writing new features.

## Get the code (WSL2, Linux filesystem)
```bash
cd ~ && git clone https://github.com/Jay14090/Dwarpal.git && cd Dwarpal
git checkout claude/sleepy-ptolemy-2xcbeh
make setup          # uv sync, .env, docker compose up (Postgres+pgvector, MediaMTX), alembic upgrade
make test && make lint
make web-install && make web-build
```
Data, caches, model weights and the DB are **not** in git (`data/`, `models/` are gitignored); you rebuild
them below.

## Do these in order (each is "held" in progress.md)
| # | Task | Commands | Expected result / where it shows |
|---|---|---|---|
| 1 | GPU sanity | `nvidia-smi`; `uv run python -c "import torch;print(torch.cuda.is_available())"` | `True`. If False on a GTX 10xx: `uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126` |
| 2 | MEVA data (5 cameras) | `make data-meva` | `data/processed/meva/*.mp4` |
| 3 | SmartSpaces scene | put `HF_TOKEN` in `.env` (accept the dataset terms on HF first), `make data-smartspaces` | `data/processed/smartspaces/` (scene_071, 6 cameras, ~1.2 GB); cameras are appended to `config/cameras.yaml`. Then choose the demo grid: 4 dataset cameras + webcam `enabled: true`, the rest `false` (8 GB VRAM) |
| 4 | Index everything (GPU) | `make index` (all cached cameras: detections + OSNet Re-ID + faces + plates). OSNet weights download from Google Drive on first run (`reid.weights_url`) | `data/cache/<cam>/{tracks,reid,faces}.npz`. The sandbox used `yolo26n`; default is `yolo26s` |
| 5 | **IDF1 + role metrics** | `make enroll-sim && make eval` | `data/eval/smartspaces_mtmc.json`, `smartspaces_identity.json` → `/metrics` page |
| 6 | Search index | `make index-db` | tracks + CLIP in Postgres; `/search` works |
| 7 | Alerts check with real Re-ID | `make events-replay CAMERAS="meva_g421 meva_g299"` | `incidents_with_more_than_one_alert: 0`; also look at how many global IDs/alerts now (sandbox used a weak fallback that merged ~160 tracks into 6 IDs) |
| 8 | Demo + webcam | WSL: `make demo`; Windows PowerShell: `.\scripts\webcam_publish.ps1` | Follow `docs/demo_script.md`; the unknown → enroll → resident flip is the P4 DoD |
| 9 | Plates (optional data) | labelled set in `data/raw/plates_eval/<name>/` (`images/` + `labels.csv`: `filename,plate[,x1,y1,x2,y2]`), `make plates-eval PLATES=data/raw/plates_eval/<name>` | exact-plate accuracy on `/metrics` |
| 10 | Gate clips | copy clips to `data/raw/gate_vehicles/`, `make gate`, register plates on `/registry` | registered / unregistered plate events |
| 11 | LLM search (optional) | `.env`: `LLM_PROVIDER=anthropic`, `LLM_API_KEY=…` (or `openai` + `LLM_BASE_URL`) | `/search` shows "parsed by llm:anthropic"; falls back to rules on any error |

After each: record the real numbers in `docs/progress.md` (the phase's Status section and the "Where things
stand" table at the end), never estimates.

## Things that were only verified in the sandbox, so watch for them locally
- **Never run on a GPU**: FP16 paths (`quantize=16` in Ultralytics, CLIP/OSNet half), CUDA memory with 5
  cameras under 8 GB. If VRAM runs out: keep dataset cameras `cached`, lower `pipeline.realtime_max_fps`, or
  use `yolo26n.pt`.
- **InsightFace (`buffalo_l`) on the live webcam** was only unit-tested; thresholds `identity.t_face` (0.45),
  `unknown_after_observations` (8), `face.min_person_height_px` (140) may need tuning on your camera.
- **OSNet weights never loaded** (Drive blocked); code was tested with random checkpoints. `global_tracker`
  thresholds (`match_threshold` 0.55 etc.) are untuned. `make eval EVAL_ARGS="--sweep match_threshold=0.4,0.5,0.6"` compares
  settings on SmartSpaces.
- **Clip playback in a real browser** (H.264 from ffmpeg, verified with ffprobe only).
- **Hugging Face CLIP weights**: the sandbox fell back to the OpenCLIP GitHub weights; locally HF should work.
- Engine start-up took ~100 s in the sandbox because blocked downloads timed out; locally it should be fast.

## Known limitations (by design, documented)
- MEVA outdoor people are 30–47 px tall → never identified (stay *pending*) → no person alerts there unless
  a rule lists `pending` in `params.roles` (`config/rules.yaml`).
- "Admin" for unblur is just the `X-Actor` header in `privacy.admin_actors`: demo-grade, not authentication.
- Clothing-colour accuracy was spot-checked on mostly-dark-clothing footage; re-check on SmartSpaces.

## How the code is organised (where to look)
- Spec and rules of work: `CLAUDE.md`. Decisions/log: `docs/progress.md`. Demo: `docs/demo_script.md`.
- Config (all thresholds, never hardcode): `config/settings.yaml`, `cameras.yaml`, `rules.yaml`;
  loaded/validated by `backend/app/core/config.py` (strict pydantic; env overrides in `ENV_OVERRIDES`).
- Engine (separate process): `backend/app/pipeline/engine.py` → per-camera `worker.py` → hooks in order
  `crosscam.py` (global IDs) → `person_hooks.py` (faces, identity) → `rules.py` → `anpr/hook.py` →
  `indexer.py`. API ↔ engine via queues (`EngineProcess.request(cmd, …)`).
- API: `backend/app/main.py` + `backend/app/api/*.py`. Frontend: `frontend/src/app/<page>/page.tsx`.
- Offline scripts: `scripts/` (`index_cameras.py`, `index_tracks.py`, `evaluate.py`, `replay_events.py`, …).

## Conventions to keep
- `make lint` (ruff check + format) and `make test` green before every commit; `make web-build` for frontend.
- Conventional commits per phase/task (`feat(p4): …`, `fix(reid): …`).
- Every metric from a real run, saved under `data/eval/` (the `/metrics` page reads only those files).
- Note any new decision in `docs/progress.md` (next number: **D57**).

## Paste this as your first message to the new Claude Code session
> Read CLAUDE.md, docs/handoff.md and docs/progress.md. All phases are built; we're on branch
> claude/sleepy-ptolemy-2xcbeh. Work through the table in docs/handoff.md "Do these in order" on this machine,
> starting at step 1. Run each step, fix anything that breaks, record real numbers in docs/progress.md, and
> commit after each step. Ask me only when you need something from me (HF token, webcam, data files).
