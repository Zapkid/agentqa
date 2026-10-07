"""OpenTelemetry metrics (exported via OTLP to the Collector, which exposes them to Prometheus).

Metric names follow the brief: ``agentqa_llm_tokens_total`` etc. OTel instrument names use
dots/underscores; the Collector's Prometheus exporter keeps underscores as written, and
counters get the ``_total`` suffix from the name itself.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, MetricReader
from opentelemetry.sdk.resources import Resource

_lock = threading.Lock()
_reader: InMemoryMetricReader | None = None
_instruments: dict[str, Any] = {}

COUNTERS = {
    "agentqa_llm_tokens_total": "LLM tokens by agent, provider, model and direction",
    "agentqa_llm_cost_usd_total": "LLM cost in USD by kind (actual | list_equivalent)",
    "agentqa_llm_errors_total": "Provider errors by provider and type",
    "agentqa_fallbacks_total": "Provider fallbacks by from and to",
    "agentqa_cache_hits_total": "LLM disk cache hits",
    "agentqa_guardrail_events_total": "Guardrail events by guardrail and action",
    "agentqa_findings_total": "Findings by class and severity",
    "agentqa_tasks_total": "Delegated tasks by tier, task_type and outcome",
    "agentqa_escalations_total": "Tier escalations by from_tier, to_tier and reason",
    "agentqa_tokens_saved_estimate_total": "Estimated tokens saved vs naive strong-only baseline",
    "agentqa_dedupe_removed_total": "Intents removed by dedupe",
}
HISTOGRAMS = {
    "agentqa_llm_latency_seconds": "LLM call latency",
    "agentqa_run_duration_seconds": "Pipeline run duration",
    "agentqa_batch_size": "Items per batched LLM call",
}
GAUGES = {
    "agentqa_eval_recall": "Bug recall of the latest eval",
    "agentqa_eval_precision": "Precision of the latest eval",
    "agentqa_eval_triage_accuracy": "Triage accuracy of the latest eval",
    "agentqa_eval_hallucination_rate": "Hallucination rate of the latest eval",
    "agentqa_coverage_ratio": "Covered intents / planned intents by risk",
    "agentqa_incremental_reuse_ratio": "Share of tests reused by an incremental run",
    "agentqa_perf_latency_seconds": "Load-run latency by scenario, endpoint and quantile",
    "agentqa_perf_rps": "Load-run throughput",
    "agentqa_perf_error_ratio": "Load-run error ratio",
    "agentqa_perf_slo_verdict": "1 = SLO met, 0 = violated",
}


def setup_metrics() -> None:
    global _reader
    with _lock:
        if _reader is not None:
            return
        _reader = InMemoryMetricReader()
        readers: list[MetricReader] = [_reader]
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if os.environ.get("AGENTQA_TRACING", "0") == "1" and endpoint:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            readers.append(
                PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=f"{endpoint.rstrip('/')}/v1/metrics"),
                    export_interval_millis=5000,
                )
            )
        resource = Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "agentqa")})
        metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=readers))
        meter = metrics.get_meter("agentqa")
        for name, desc in COUNTERS.items():
            _instruments[name] = meter.create_counter(name, description=desc)
        for name, desc in HISTOGRAMS.items():
            _instruments[name] = meter.create_histogram(name, description=desc)
        for name, desc in GAUGES.items():
            _instruments[name] = meter.create_gauge(name, description=desc)


def inc(name: str, value: float = 1, **labels: Any) -> None:
    setup_metrics()
    _instruments[name].add(value, {k: str(v) for k, v in labels.items()})


def observe(name: str, value: float, **labels: Any) -> None:
    setup_metrics()
    _instruments[name].record(value, {k: str(v) for k, v in labels.items()})


def gauge(name: str, value: float, **labels: Any) -> None:
    setup_metrics()
    _instruments[name].set(value, {k: str(v) for k, v in labels.items()})


def snapshot() -> dict[str, list[tuple[dict[str, Any], float]]]:
    """Current values from the in-memory reader: {metric: [(labels, value), ...]}.

    Counters report their cumulative sum, histograms their sum, gauges their last value.
    """
    setup_metrics()
    assert _reader is not None
    data = _reader.get_metrics_data()
    out: dict[str, list[tuple[dict[str, Any], float]]] = {}
    if data is None:
        return out
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                points = []
                for dp in metric.data.data_points:
                    value: Any = getattr(dp, "value", None)
                    if value is None:
                        value = getattr(dp, "sum", 0.0)
                    points.append((dict(dp.attributes or {}), float(value or 0.0)))
                out[metric.name] = points
    return out


def total(name: str, **match: Any) -> float:
    """Sum of a metric across label sets that match ``match``."""
    result = 0.0
    for labels, value in snapshot().get(name, []):
        if all(str(labels.get(k)) == str(v) for k, v in match.items()):
            result += value
    return result
