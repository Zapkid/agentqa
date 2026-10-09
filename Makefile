# AgentQA developer entry points. Everything runs through uv.
UV ?= uv
PY := $(UV) run python
export PYTHONUNBUFFERED=1

.PHONY: install lint fmt typecheck test test-fast cov secrets audit demo up down eval eval-live eval-replay perf-ab target example-tasks veroniqa veroniqa-hosted videos video-intro clean

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

# Known-vulnerability scan of the locked dependencies. The ignored IDs are chromadb *server*
# advisories that do not apply to the embedded client we use (docs/SECURITY.md); anything new fails.
AUDIT_IGNORES := --ignore-vuln PYSEC-2026-311 --ignore-vuln PYSEC-2026-3813 \
                 --ignore-vuln PYSEC-2026-3814 --ignore-vuln PYSEC-2026-3815
audit:
	@mkdir -p .cache
	@$(UV) export --frozen --no-hashes --no-emit-project -q -o .cache/requirements.txt
	uvx pip-audit==2.10.1 -r .cache/requirements.txt --no-deps --disable-pip $(AUDIT_IGNORES)

# Run the target API locally (no Docker). BUGS=B01,B04 PERF_BUGS=P01 make target
target:
	$(UV) run uvicorn target_api.app.main:app --port 8000

# VeroniQA: chat, knowledge base and test runs in the browser (http://127.0.0.1:8501).
veroniqa:
	$(UV) run agentqa veroniqa

# The public-demo build (docs/DEPLOY.md): the introduction video as the home page and the app at
# /talk/ (private workspaces, simulated models), on http://127.0.0.1:8502.
veroniqa-hosted:
	VERONIQA_HOSTED=1 AGENTQA_HOME=.agentqa/hosted $(UV) run uvicorn agentqa.veroniqa.server:app --host 127.0.0.1 --port 8502

# Promo videos (media/videos/*.mp4), rendered from results/*.json and media/veroniqa/.
# Refresh the VeroniQA screenshots first with:
#   uv run --with playwright==1.55.0 python scripts/videos/capture_veroniqa.py
videos:
	$(PY) scripts/videos/render.py

# Narrated 1080p introduction to VeroniQA (voice: Kokoro-82M, run locally; music: generated).
# The second command checks the soundtrack with an independent speech recogniser.
TTS_DEPS := --with kokoro-onnx==0.6.1 --with soundfile==0.14.0
video-intro:
	$(UV) run $(TTS_DEPS) python scripts/videos/veroniqa_intro.py
	$(UV) run $(TTS_DEPS) --with sherpa-onnx==1.13.8 python scripts/videos/check_voice.py

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
