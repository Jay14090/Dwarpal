# Dwarpal developer commands. Run from WSL2 (or any Linux/macOS shell).
SHELL   := /bin/bash
UV      ?= uv
COMPOSE ?= docker compose
RUN     := $(UV) run
HOST    ?= 0.0.0.0
PORT    ?= 8000

.DEFAULT_GOAL := help
.PHONY: help setup env infra-up infra-down infra-logs migrate dev test lint fmt \
        data data-meva data-smartspaces inspect streams streams-stop index index-db search-check events-replay web web-install web-build enroll-sim gate plates-eval demo eval clean

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

setup: env ## Install Python deps, start Postgres + MediaMTX, run migrations
	$(UV) sync
	$(MAKE) infra-up
	$(MAKE) migrate

env: ## Create .env from .env.example if missing
	@[ -f .env ] || { cp .env.example .env && echo "created .env from .env.example"; }

infra-up: env ## Start Postgres+pgvector and MediaMTX, wait for the DB
	$(COMPOSE) up -d
	$(RUN) python scripts/wait_for_db.py --timeout 90

infra-down: ## Stop infra containers (data volume kept)
	$(COMPOSE) down

infra-logs: ## Tail infra logs
	$(COMPOSE) logs -f --tail=50

migrate: ## Apply Alembic migrations
	$(RUN) alembic upgrade head

dev: infra-up ## Run the API with auto-reload on http://localhost:$(PORT)
	$(RUN) alembic upgrade head
	$(RUN) uvicorn app.main:app --reload --host $(HOST) --port $(PORT) \
		--reload-dir backend/app --reload-dir config

web-install: ## Install the dashboard's npm dependencies (frontend/)
	cd frontend && npm install

web: ## Run the Next.js dashboard on http://localhost:3000 (needs `make dev` for the API)
	cd frontend && npm run dev

web-build: ## Typecheck, lint and build the dashboard
	cd frontend && npx tsc --noEmit && npm run lint && npm run build

test: ## Run the test suite (DB tests skip if Postgres is down)
	$(RUN) pytest

lint: ## Ruff lint + format check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

fmt: ## Auto-fix lint and format
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

data: data-meva data-smartspaces ## Download + adapt both datasets (lists sizes first)

data-meva: ## Download the MEVA subset (~0.5 GB) and convert to data/processed/meva
	scripts/download_meva.sh
	$(RUN) python scripts/adapt_meva.py

data-smartspaces: ## Download the SmartSpaces scene subset (~1.2 GB) and convert it
	$(RUN) python scripts/download_smartspaces.py
	$(RUN) python scripts/adapt_smartspaces.py
	$(RUN) python scripts/register_cameras.py smartspaces

inspect: ## Print cameras, durations and identity counts of processed datasets
	$(RUN) python scripts/inspect_dataset.py

streams: infra-up ## Loop every processed video as RTSP (rtsp://localhost:8554/<camera>)
	scripts/stream_cameras.sh start

streams-stop: ## Stop the RTSP loopers
	scripts/stream_cameras.sh stop

index: ## Precompute tracked detections for cached-mode cameras (or run notebooks/colab_index.ipynb)
	$(RUN) python scripts/index_cameras.py $(CAMERAS)

index-db: ## Index cached cameras into Postgres for search: colours, height, CLIP (after make index)
	$(RUN) python scripts/index_tracks.py $(if $(CAMERAS),--cameras $(CAMERAS)) --replace

search-check: ## Parse every query in tests/search_queries.yaml and compare with the expected filters
	$(RUN) pytest backend/tests/test_search.py -q -k rule_parser

events-replay: ## Replay cached cameras through identity + rules; checks one alert per incident
	$(RUN) python scripts/replay_events.py $(if $(CAMERAS),--cameras $(CAMERAS)) $(REPLAY_ARGS)

demo: ## Start everything for the demo: infra, API + engine, dashboard (Ctrl-C stops all)
	./scripts/demo.sh

enroll-sim: ## Simulated enrollment on SmartSpaces (15 residents, 5 staff, seed 42)
	$(RUN) python scripts/simulate_enrollment.py --residents 15 --staff 5 --seed 42 $(ENROLL_ARGS)

gate: ## Turn your clips in data/raw/gate_vehicles/ into ANPR gate cameras
	$(RUN) python scripts/adapt_gate_clips.py

plates-eval: ## Plate OCR/detector metrics: PLATES=<dataset dir> (or synthetic sanity check)
	$(RUN) python scripts/eval_plates.py $(if $(PLATES),--dataset-dir $(PLATES),--synthetic 500)

eval: ## Compute metrics against ground truth (IDF1 etc.; needs SmartSpaces + make index)
	$(RUN) python scripts/evaluate.py $(EVAL_ARGS)

clean: ## Remove caches (keeps data/ and the DB volume)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
