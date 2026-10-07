# ADR 0005: Tracing conventions (GenAI semconv + OpenInference + agentqa.*)

Status: accepted (2026-10-07)

## Decision
- Span tree `run` > `stage` > `agent` > `llm_call` / `tool_call` (plus `guard` and `retrieval`).
  Worker threads inherit the trace context (`contextvars.copy_context()` per task), so one run is
  one trace.
- LLM spans carry the OpenTelemetry GenAI attributes (`gen_ai.provider.name`,
  `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`,
  `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`,
  `gen_ai.response.finish_reasons`; names imported from the installed `opentelemetry-semconv`, so
  they are not typed from memory), OpenInference attributes (`openinference.span.kind`,
  `llm.model_name`, `llm.token_count.*`, `input.value`/`output.value`) so Phoenix renders them as
  LLM spans, and `agentqa.*` attributes: prompt name/version/hash, cache status, retry count, cost
  actual and list-equivalent, fallback source, tier, task id.
- Guardrail decisions, validation failures, fallbacks, escalations and delegations are span
  **events**, not log lines, so they sit on the timeline where they happened.
- Spans always go to an in-memory exporter as well, and each run writes `spans.jsonl`. The run
  page and the tests read that file; nothing requires Phoenix to be up.

## Consequences
From one `trace_id` you can walk from a finding (triage `llm_call`, cluster id) back through the
generator call that wrote the test, its tool calls and retrieved chunk ids, to the planner call
and the run, with cost on every LLM span (tested in `tests/test_pipeline_e2e.py`).
