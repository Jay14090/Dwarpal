# Dwarpal

AI security operator for gated communities: multi-camera person tracking with cross-camera
global IDs, resident/staff/unknown labeling, Indian ANPR, real-time alerts and natural-language
footage search, built privacy-first.

> Status: **P0-P1** complete (see progress log for what is held). See [`docs/progress.md`](docs/progress.md) for the phase log
> and decisions, and [`CLAUDE.md`](CLAUDE.md) for the full spec.

## Prerequisites (Windows 11 + WSL2)

Run every command below **inside WSL2 (Ubuntu)**, in a clone stored on the Linux filesystem
(e.g. `~/Dwarpal`, not `/mnt/c/...`; it is much faster).

1. Docker Desktop with *Settings → Resources → WSL integration* enabled for your distro.
2. `make`, `git`, `ffmpeg`: `sudo apt install -y make git ffmpeg`
3. `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh` (uv fetches Python 3.11 itself)
4. NVIDIA: install the current Windows NVIDIA driver only. Do **not** install a Linux driver in
   WSL; CUDA reaches WSL through the Windows driver. Check with `nvidia-smi` inside WSL.

## Quickstart

```bash
make setup     # uv sync, create .env, start Postgres+pgvector and MediaMTX, run migrations
make dev       # API on http://localhost:8000 (auto-reload)
curl localhost:8000/health
make test      # pytest (DB tests skip automatically if Postgres is down)
make lint
make help      # all targets
```

`/health` returns `"status": "ok"` when the DB is reachable, pgvector is installed and
migrations are applied; otherwise `"degraded"` with details.

## Data (P1)

```bash
make data-meva          # MEVA outdoor subset: ~0.5 GB from the public S3 bucket (needs `aws` CLI)
make data-smartspaces   # SmartSpaces retail scene: ~1.2 GB from Hugging Face (HF_TOKEN if gated)
make inspect            # cameras, durations, identity counts
uv run python scripts/render_gt_overlay.py meva meva_g336   # 10 s GT overlay -> data/clips/
make streams            # loop every processed video as rtsp://localhost:8554/<camera>
```

Open any stream in VLC: *Media → Open Network Stream → `rtsp://localhost:8554/meva_g336`*.

**Webcam on Windows + WSL2:** WSL2 cannot see USB webcams, so publish it from Windows:
`winget install Gyan.FFmpeg`, then in PowerShell from the repo folder
`.\scripts\webcam_publish.ps1` (or `-List` to choose a camera). The backend reads `rtsp://localhost:8554/webcam`.

**GPU note:** the PyPI `torch` wheels target CUDA 13, which needs a recent Windows NVIDIA driver
(R580+) and a GTX 16xx/RTX-class GPU. For a GTX 10xx (Pascal) card, run
`uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126`.

## Layout

```
config/          settings.yaml (all thresholds/paths), cameras.yaml, rules.yaml, mediamtx.yml
backend/app/     FastAPI app: core/ (config, device), db/ (models, session), api/
backend/alembic/ migrations
backend/tests/   pytest
scripts/         dataset, streaming and evaluation tools
docs/            progress.md, demo_script.md
data/            local only (gitignored)
```

## Services (docker-compose)

| Service | Port | Purpose |
|---|---|---|
| Postgres 16 + pgvector | 5432 | tracks, embeddings, plate reads, events, audit log |
| MediaMTX | 8554 (RTSP/TCP), 8888 (HLS), 9997 (API) | simulated CCTV cameras and the webcam feed |

Ports are configurable in `.env`.

## Licensing notes

- InsightFace pretrained models (`buffalo_l`), used for face recognition from P4, are licensed
  for **non-commercial research use only**.
- NVIDIA PhysicalAI-SmartSpaces is CC-BY-4.0. MEVA and Indian-LPR follow their own terms.
