# AgentQA build plan

This plan follows the milestones in the brief (section 12). Every milestone ends with green
`make lint test` and a commit. The plan states the environment constraints up front, because
they decide what can be measured in this build and what must wait for a run with real keys.

## Constraints of the build environment (and how the plan handles them)

| Constraint | Consequence | Handling |
| --- | --- | --- |
| No Docker daemon in the build container | `docker compose` cannot be exercised here | Compose files, Collector, Prometheus and Grafana provisioning are written and statically validated (YAML/JSON parse, config lint). The pipeline itself never *requires* Docker: the target API is launched as a local `uvicorn` subprocess per build (clean / single-bug / all-bugs), which is also what CI uses. |
| No LLM API keys in the build container | No live model numbers can be produced here | Real adapters (Anthropic, Gemini, OpenRouter) are implemented against current SDKs and unit-tested with mocked transports. All measured numbers in this repo come from the deterministic **simulated** adapter and are labelled as such. `make eval-live` produces the real-model table once keys are present. Recorded in `docs/LIMITATIONS.md`. |
| Hugging Face blocked (no embedding model download) | `fastembed` cannot fetch weights here | `Embedder` interface with `fastembed` as the default when available and a deterministic local hashing embedder as the fallback (used in CI). ADR 0004. |
| Google AI docs blocked from this container | Gemini field names cannot be read from the docs site | Verified against the installed `google-genai` SDK (`GenerateContentConfig.response_json_schema`, `usage_metadata.cached_content_token_count`). Model IDs carry an "as of" date and a `verify_before_use` flag. |

## Milestones and task breakdown

### M1. Scaffold and provider layer
- `pyproject.toml` (uv, Python 3.12), ruff, mypy, pytest, coverage, Makefile, `.env.example`, `.gitignore`, pre-commit with a local secret scan.
- `config/*.yaml` with pydantic loaders: providers (rpm/tpm/rpd), models (profiles, tiers, judge family rule), pricing (as-of dates).
- `agentqa/llm`: `LLMClient` protocol, `LLMResult`, normalised tool calls; adapters for Anthropic, Gemini, OpenRouter; `FakeLLM` (scripted) and `SimulatedLLM` (deterministic "skill" model for pipeline tests).
- Token-bucket rate limiter, backoff with jitter, disk cache (`off` / `read_write` / `replay_only`), structured-output enforcement with feedback retries, fallback chain with circuit breaker, cost ledger (actual and list-equivalent, cached input tokens).
- `agentqa/obs`: tracing skeleton (run > stage > agent > llm_call/tool_call), metrics, structlog with redaction.
- Prompt registry (`prompts/*.md` with semver front matter, hashed into cache key).

### M2. Target API and benchmark truth
- `target_api`: Orders and Invoicing API (FastAPI + SQLite, three roles, token auth), OTel spans incl. DB query spans, `/__seed`, `/__metrics` (process RSS, query counts).
- `bugs.yaml` (B01..B12), `perf_bugs.yaml` (P01..P06), toggled by `BUGS` / `PERF_BUGS`.
- Customer-style requirement docs with SLOs, `poisoned.md`.
- Hand-written proofs: one failing test per functional bug and one measurement per perf defect.

### M3. Ingest, planner, Tier-0 synthesizer, generator, grounding guard
- OpenAPI 3.x parser to `Endpoint`, chunker, Chroma store (idempotent upserts by content hash).
- Planner (`TestPlan` of `TestIntent`s, cited source chunks, per-endpoint cap).
- Tier-0 synthesizer: contract, required-field, type, enum/boundary, pagination, missing-auth tests from the spec.
- Generator with tools (`get_endpoint_schema`, `search_docs`, `list_fixtures`, `get_example_payload`), static checks (ast, ruff, import allowlist, banned calls).
- Grounding guard over `requests_made` and AST request calls; repair once, then quarantine; `hallucination_rate`.

### M4. Executor, triage, report
- Sandboxed pytest subprocess: timeout, rlimits, network allowlist and method guard enforced by an injected pytest plugin, request/response log (redacted), JSON results, server log slice, 3x flaky reruns.
- Cluster-first triage (deterministic clustering, one LLM call per cluster), evidence links, `needs_review` downgrade, curl repro.
- Reporter: Markdown + self-contained HTML, executive summary with an explicitly labelled effort estimate; LLM judge (different family) for report quality.

### M5. Supervisor and dispatcher (part 1)
- Typed task DAG, fan-out/fan-in, priorities, failure isolation (uncovered with reason), SQLite checkpoint and resume.
- Tiers T0/T1/T2, transparent difficulty features, verification-gated cascade (repair once, escalate with feedback), delegation ledger (SQLite + span events), static role routing baseline.

### M6. Performance agent
- `WorkloadSpec` (LLM, validated), deterministic Locust renderer, load shapes (smoke/load/stress/spike/soak), load guard, runner with warm-up discard and ≥3 iterations.
- Perf triage: knee-point detection, SLO verdicts, noise band, bottleneck classifier over a compact evidence bundle, A/B comparison, matplotlib charts.

### M7. Dispatcher (part 2)
- Learned routing stats (Thompson sampling with a strong-tier floor), batching with per-item validation, embedding dedupe, diff-aware incremental runs, budget-aware allocation, uncovered-with-reason reporting.

### M8. Observability
- `deploy/`: docker-compose (target, Phoenix, Collector, Prometheus, Grafana, cAdvisor), Collector config, Prometheus scrape, provisioned dashboards (cost, tokens, latency, errors, fallbacks, cache, guardrails, findings, perf, delegation) and alert rules with a webhook contact point.
- Log/trace correlation (`trace_id`, `run_id`).

### M9. Evals and CI
- Differential ground truth (clean, single-bug, all-bugs builds), metrics (recall, precision, triage accuracy, validity, hallucination, flakiness, cost per bug), perf A/B ground truth with a measured noise band.
- Profile matrix, strategy comparison S0–S3, ablations, `RESULTS.md` and charts, replay gate vs `evals/baselines/main.json`, cost gate.
- Injection test plus an adversarial set; judge calibration set.
- `.github/workflows/ci.yml` (lint, types, tests, replay gate, perf A/B smoke) and `eval-live.yml` (nightly / dispatch).

### M10. MCP, API/UI, memory
- MCP server (stdio + streamable HTTP) with `ingest_spec`, `plan_tests`, `run_suite`, `triage_run`, `get_report`, `list_runs`, perf tools, report resources.
- FastAPI + Jinja run history, run detail, scoreboard.
- Lessons store (SQLite) retrieved into planner/generator prompts; within-run scratchpad.

### M11. Polish
- README (problem, architecture diagram, how to run, results with date and commit, guardrails, observability, eval method and limits, limitations, enterprise next steps), `DEMO.md`, ADRs, `LIMITATIONS.md`, `ENTERPRISE.md`.

## Definition of done per milestone
`make lint` (ruff + mypy) and `make test` pass; new code has unit tests; commit pushed to the
working branch. Anything deferred or not implemented is listed in `docs/LIMITATIONS.md` the
moment it is deferred, not at the end.

## Status (end of the first build pass, 2026-10-07)

| milestone | status | evidence |
|---|---|---|
| M1 scaffold + provider layer | done | `tests/test_llm_core.py`, `tests/test_adapters.py` (stub clients; no live calls, no keys) |
| M2 target + benchmark truth | done | `target_api/tests/` (36 functional proofs, 6 perf measurements) |
| M3 ingest, planner, T0, generator, grounding | done | `tests/test_ingest.py`, `tests/test_generation_guards.py` |
| M4 executor, triage, report | done | `tests/test_executor.py`, `tests/test_triage_report.py`, `tests/test_pipeline_e2e.py` |
| M5 supervisor + dispatcher part 1 | done | `tests/test_orchestrator_dispatch.py` |
| M6 performance agent | done | `tests/test_perf.py`, `tests/test_perf_e2e.py` |
| M7 dispatcher part 2 | done (self-consistency voting not implemented) | strategy/ablation results |
| M8 observability | code + config done; stack not run (no Docker) | `deploy/`, ADR 0005 |
| M9 evals + CI | done with simulated models; live matrix pending keys | `RESULTS.md`, `.github/workflows/` |
| M10 MCP, UI, memory | done (no GUI-client screenshot) | `tests/test_mcp_server.py`, `tests/test_api.py` |
| M11 polish | done; demo GIF and dashboard screenshots pending a machine with Docker | README, DEMO, LIMITATIONS, ENTERPRISE |
