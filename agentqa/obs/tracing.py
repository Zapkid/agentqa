"""Tracing setup and span helpers.

Span tree: ``run`` > ``stage`` > ``agent`` > ``llm_call`` / ``tool_call``.

Spans always go to an in-memory exporter (so every run can persist ``spans.jsonl`` and the
run page can show the trace without any infrastructure). When ``AGENTQA_TRACING=1`` and an
OTLP endpoint is configured, spans are also exported to the Collector, which forwards them to
Phoenix. LLM spans carry the OpenTelemetry GenAI semantic-convention attributes (names taken
from ``opentelemetry.semconv``) plus OpenInference attributes so Phoenix renders them as LLM
spans, plus ``agentqa.*`` custom attributes.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as genai
from opentelemetry.trace import Span, Status, StatusCode

# GenAI semantic-convention attribute names (verified against the installed semconv package).
GEN_AI_OPERATION = genai.GEN_AI_OPERATION_NAME
GEN_AI_PROVIDER = genai.GEN_AI_PROVIDER_NAME
GEN_AI_REQUEST_MODEL = genai.GEN_AI_REQUEST_MODEL
GEN_AI_RESPONSE_MODEL = genai.GEN_AI_RESPONSE_MODEL
GEN_AI_INPUT_TOKENS = genai.GEN_AI_USAGE_INPUT_TOKENS
GEN_AI_OUTPUT_TOKENS = genai.GEN_AI_USAGE_OUTPUT_TOKENS
GEN_AI_CACHE_READ_TOKENS = genai.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS
GEN_AI_FINISH_REASONS = genai.GEN_AI_RESPONSE_FINISH_REASONS

_lock = threading.Lock()
_memory_exporter: InMemorySpanExporter | None = None
_provider: TracerProvider | None = None


def setup_tracing(service_name: str = "agentqa") -> TracerProvider:
    """Install the tracer provider once per process. Safe to call repeatedly."""
    global _memory_exporter, _provider
    with _lock:
        if _provider is not None:
            return _provider
        resource = Resource.create(
            {"service.name": os.environ.get("OTEL_SERVICE_NAME", service_name)}
        )
        provider = TracerProvider(resource=resource)
        _memory_exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(_memory_exporter))
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if os.environ.get("AGENTQA_TRACING", "0") == "1" and endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces"))
            )
        trace.set_tracer_provider(provider)
        _provider = provider
        return provider


def tracer() -> trace.Tracer:
    setup_tracing()
    return trace.get_tracer("agentqa")


def _clean(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(v, (str, bool, int, float)) for v in value
    ):
        return list(value)
    return json.dumps(value, default=str)[:4000]


@contextmanager
def span(name: str, kind: str, **attrs: Any) -> Iterator[Span]:
    """Open a span of an agentqa ``kind`` (run | stage | agent | llm_call | tool_call | guard)."""
    openinference_kind = {
        "run": "CHAIN",
        "stage": "CHAIN",
        "agent": "AGENT",
        "llm_call": "LLM",
        "tool_call": "TOOL",
        "guard": "GUARDRAIL",
        "retrieval": "RETRIEVER",
    }.get(kind, "CHAIN")
    with tracer().start_as_current_span(name) as sp:
        sp.set_attribute("agentqa.span_kind", kind)
        sp.set_attribute("openinference.span.kind", openinference_kind)
        for key, value in attrs.items():
            if value is not None:
                sp.set_attribute(key, _clean(value))
        try:
            yield sp
        except Exception as exc:
            sp.record_exception(exc)
            sp.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
            raise


def event(name: str, **attrs: Any) -> None:
    """Add an event (guardrail decision, fallback, validation failure...) to the current span."""
    current = trace.get_current_span()
    current.add_event(name, {k: _clean(v) for k, v in attrs.items() if v is not None})


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x")


def _span_to_dict(sp: ReadableSpan) -> dict[str, Any]:
    ctx = sp.get_span_context()
    parent = sp.parent
    return {
        "name": sp.name,
        "trace_id": format(ctx.trace_id, "032x") if ctx else None,
        "span_id": format(ctx.span_id, "016x") if ctx else None,
        "parent_id": format(parent.span_id, "016x") if parent else None,
        "start_ns": sp.start_time,
        "end_ns": sp.end_time,
        "status": sp.status.status_code.name,
        "attributes": dict(sp.attributes or {}),
        "events": [
            {"name": e.name, "ts_ns": e.timestamp, "attributes": dict(e.attributes or {})}
            for e in sp.events
        ],
    }


def finished_spans(trace_id: str | None = None) -> list[dict[str, Any]]:
    setup_tracing()
    assert _memory_exporter is not None
    spans = [_span_to_dict(s) for s in _memory_exporter.get_finished_spans()]
    if trace_id:
        spans = [s for s in spans if s["trace_id"] == trace_id]
    return spans


def dump_spans(path: Path, trace_id: str | None = None) -> int:
    spans = finished_spans(trace_id)
    with open(path, "w", encoding="utf-8") as fh:
        for s in spans:
            fh.write(json.dumps(s, default=str) + "\n")
    return len(spans)


def reset_memory_spans() -> None:
    setup_tracing()
    assert _memory_exporter is not None
    _memory_exporter.clear()


def flush() -> None:
    if _provider is not None:
        _provider.force_flush()
