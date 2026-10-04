# Dwarpal developer commands. Run from WSL2 (or any Linux/macOS shell).
SHELL   := /bin/bash
UV      ?= uv
COMPOSE ?= docker compose
RUN     := $(UV) run
HOST    ?= 0.0.0.0
PORT    ?= 8000

.DEFAULT_GOAL := help
.PHONY: help setup env infra-up infra-down infra-logs migrate dev test lint fmt \
        data data-meva data-smartspaces inspect streams streams-stop index demo eval clean

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

index: ## (P2/P3) Precompute detections for cached-mode cameras
	@echo "make index lands in P2/P3"; exit 1

demo: ## (P11) Start everything for the demo
	@echo "make demo lands in P11"; exit 1

eval: ## (P3+) Compute metrics against ground truth
	@echo "make eval lands in P3"; exit 1

clean: ## Remove caches (keeps data/ and the DB volume)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
