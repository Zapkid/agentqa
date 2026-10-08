# AgentQA developer entry points. Everything runs through uv.
UV ?= uv
PY := $(UV) run python
export PYTHONUNBUFFERED=1

.PHONY: install lint fmt typecheck test test-fast cov secrets demo up down eval eval-live eval-replay perf-ab target example-tasks clean

install:
	$(UV) sync

lint:
	$(UV) run ruff check .
	$(UV) run mypy

fmt:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest

test-fast:
	$(UV) run pytest -m "not slow"

cov:
	$(UV) run pytest --cov --cov-report=term-missing

secrets:
	$(PY) scripts/secret_scan.py

# Run the target API locally (no Docker). BUGS=B01,B04 PERF_BUGS=P01 make target
target:
	$(UV) run uvicorn target_api.app.main:app --port 8000

# Second, differently shaped target: Swagger 2.0 contract, X-API-Key auth. BUGS=T01,T02 make example-tasks
example-tasks:
	$(UV) run uvicorn examples.tasks_api.app:app --port 8001

# Full stack: target API, Phoenix, Collector, Prometheus, Grafana, cAdvisor.
up:
	docker compose -f deploy/docker-compose.yml up -d --build

down:
	docker compose -f deploy/docker-compose.yml down

demo:
	$(PY) scripts/demo.py

eval:
	$(UV) run agentqa eval --matrix --profiles simulated

eval-replay:
	AGENTQA_CACHE_MODE=replay_only $(UV) run agentqa eval --gate

eval-live:
	$(UV) run agentqa eval --matrix --profiles free,mixed,premium

perf-ab:
	$(UV) run agentqa perf ab --type smoke

clean:
	rm -rf .agentqa runs .cache .mypy_cache .ruff_cache .pytest_cache
