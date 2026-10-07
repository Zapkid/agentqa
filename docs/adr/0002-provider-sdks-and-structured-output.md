# ADR 0002: Official SDK per provider, one normalised client

Status: accepted (2026-10-07)

## Decision
- Anthropic: official `anthropic` SDK; structured output via `output_config.format`
  (json_schema); prompt caching via `cache_control` on the static system prefix; models that
  reject sampling parameters (Opus 5.5, Sonnet 5.5) get `output_config.effort` instead of
  `temperature`; server-side refusal fallback beta enabled on those models.
- Gemini: official `google-genai` SDK; `response_mime_type=application/json` +
  `response_json_schema`; implicit context caching (reported as `cached_content_token_count`);
  automatic function calling disabled.
- OpenRouter: `openai` SDK pointed at `https://openrouter.ai/api/v1`; `response_format`
  json_schema where the model supports it.
- Every structured output is validated with pydantic regardless of the native mechanism; a
  failed validation is fed back to the model up to `structured_output.max_repair_retries`
  times, and each retry is counted (`agentqa.retry_count`).
- The SDKs' own retries are disabled (`max_retries=0`); the router owns retries, backoff,
  fallback and the circuit breaker so all of it is visible in traces and metrics.

## Verification
Field names were checked against the installed SDK versions (anthropic 1.11, google-genai
2.28, openai 3.26) and against current API docs where reachable. Adapters are unit-tested with
stub clients; they have not been exercised against live endpoints from the build environment
(no keys). See `docs/LIMITATIONS.md`.
