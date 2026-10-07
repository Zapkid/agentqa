# Limitations (kept current as the build proceeds)

## Environment of this build
- **No live model results yet.** The build container had no LLM API keys. The Anthropic,
  Gemini and OpenRouter adapters are implemented against the current SDKs and unit-tested with
  stub clients, but have not been called live from here. Every number in this repository that
  was measured here comes from the deterministic simulated models (ADR 0003) and is labelled
  SIMULATED. Run `make eval-live` with keys in `.env` to produce real-model tables.
- **Docker was not available** in the build container, so `deploy/docker-compose.yml`, the
  Collector, Prometheus and Grafana provisioning are statically validated only. The pipeline
  itself does not need Docker: target builds run as local uvicorn subprocesses.
- **Gemini docs were unreachable** from the build host; Gemini model IDs and prices come from
  third-party summaries and the installed SDK, and are flagged `verify_before_use` in
  `config/pricing.yaml`.
