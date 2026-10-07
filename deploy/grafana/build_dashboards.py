"""Generate the provisioned Grafana dashboards (dashboards as code).

Run: uv run python deploy/grafana/build_dashboards.py   (writes deploy/grafana/dashboards/*.json)
Metric names match agentqa/obs/metrics.py as exported by the Collector's Prometheus exporter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT = Path(__file__).parent / "dashboards"
DS = {"type": "prometheus", "uid": "prometheus"}


class Board:
    def __init__(self, uid: str, title: str) -> None:
        self.uid, self.title = uid, title
        self.panels: list[dict[str, Any]] = []
        self.y = 0
        self.x = 0
        self.next_id = 1

    def row(self, title: str) -> None:
        if self.x:
            self.y += 8
            self.x = 0
        self.panels.append(
            {
                "id": self.next_id,
                "type": "row",
                "title": title,
                "collapsed": False,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": self.y},
                "panels": [],
            }
        )
        self.next_id += 1
        self.y += 1

    def panel(
        self,
        title: str,
        exprs: list[tuple[str, str]],
        kind: str = "timeseries",
        unit: str = "short",
        w: int = 8,
        desc: str = "",
    ) -> None:
        if self.x + w > 24:
            self.y += 8
            self.x = 0
        self.panels.append(
            {
                "id": self.next_id,
                "type": kind,
                "title": title,
                "description": desc,
                "datasource": DS,
                "gridPos": {"h": 8, "w": w, "x": self.x, "y": self.y},
                "fieldConfig": {
                    "defaults": {"unit": unit, "custom": {"lineWidth": 2, "fillOpacity": 8}},
                    "overrides": [],
                },
                "options": {
                    "legend": {"displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "multi"},
                },
                "targets": [
                    {"refId": chr(65 + i), "datasource": DS, "expr": e, "legendFormat": legend}
                    for i, (e, legend) in enumerate(exprs)
                ],
            }
        )
        self.next_id += 1
        self.x += w

    def json(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "title": self.title,
            "schemaVersion": 39,
            "version": 1,
            "editable": True,
            "time": {"from": "now-1h", "to": "now"},
            "refresh": "10s",
            "tags": ["agentqa"],
            "templating": {"list": []},
            "panels": self.panels,
        }


def overview() -> Board:
    b = Board("agentqa-overview", "AgentQA — LLM cost, quality and guardrails")
    b.row("LLM usage and cost")
    b.panel(
        "Tokens/s by agent", [("sum by (agent) (rate(agentqa_llm_tokens_total[5m]))", "{{agent}}")]
    )
    b.panel(
        "Tokens/s by model and direction",
        [
            (
                "sum by (model, direction) (rate(agentqa_llm_tokens_total[5m]))",
                "{{model}} {{direction}}",
            )
        ],
    )
    b.panel(
        "Cost (USD, cumulative)",
        [("sum by (kind) (agentqa_llm_cost_usd_total)", "{{kind}}")],
        unit="currencyUSD",
        desc="actual = what was billed (0 on free tiers); list_equivalent = at list price",
    )
    b.panel(
        "LLM latency p50 / p95",
        [
            (
                "histogram_quantile(0.5, sum by (le) (rate(agentqa_llm_latency_seconds_bucket[5m])))",
                "p50",
            ),
            (
                "histogram_quantile(0.95, sum by (le) (rate(agentqa_llm_latency_seconds_bucket[5m])))",
                "p95",
            ),
        ],
        unit="s",
    )
    b.panel(
        "Provider errors and fallbacks",
        [
            (
                "sum by (provider, type) (rate(agentqa_llm_errors_total[5m]))",
                "error {{provider}} {{type}}",
            ),
            (
                "sum by (from, to) (rate(agentqa_fallbacks_total[5m]))",
                "fallback {{from}} -> {{to}}",
            ),
        ],
    )
    b.panel(
        "Cache hit rate",
        [
            (
                "sum(rate(agentqa_cache_hits_total[5m])) / (sum(rate(agentqa_cache_hits_total[5m])) + sum(rate(agentqa_llm_latency_seconds_count[5m])))",
                "hit rate",
            )
        ],
        unit="percentunit",
    )
    b.row("Guardrails and findings")
    b.panel(
        "Guardrail events",
        [
            (
                "sum by (guardrail, action) (increase(agentqa_guardrail_events_total[15m]))",
                "{{guardrail}} {{action}}",
            )
        ],
        kind="barchart",
        w=12,
    )
    b.panel(
        "Findings by class and severity",
        [("sum by (class, severity) (agentqa_findings_total)", "{{class}} {{severity}}")],
        kind="barchart",
        w=12,
    )
    b.row("Delegation and savings")
    b.panel(
        "Tasks by tier",
        [("sum by (tier) (increase(agentqa_tasks_total[1h]))", "{{tier}}")],
        kind="barchart",
    )
    b.panel(
        "Escalations by reason",
        [
            (
                "sum by (from_tier, to_tier, reason) (increase(agentqa_escalations_total[1h]))",
                "{{from_tier}}->{{to_tier}} {{reason}}",
            )
        ],
        kind="barchart",
    )
    b.panel(
        "Escalation rate",
        [
            (
                'sum(increase(agentqa_escalations_total[1h])) / sum(increase(agentqa_tasks_total{tier!="T0"}[1h]))',
                "escalations / LLM tasks",
            )
        ],
        unit="percentunit",
    )
    b.panel(
        "Estimated tokens saved vs strong-only",
        [("sum(agentqa_tokens_saved_estimate_total)", "tokens saved (estimate)")],
        kind="stat",
    )
    b.panel("Coverage by risk", [("agentqa_coverage_ratio", "risk {{risk}}")], unit="percentunit")
    b.panel(
        "Batch size / dedupe / incremental reuse",
        [
            (
                "histogram_quantile(0.5, sum by (le) (rate(agentqa_batch_size_bucket[1h])))",
                "median batch size",
            ),
            ("sum(agentqa_dedupe_removed_total)", "intents removed by dedupe"),
            ("agentqa_incremental_reuse_ratio", "incremental reuse ratio"),
        ],
    )
    b.row("Eval trend")
    b.panel(
        "Recall / precision / triage accuracy by commit",
        [
            ("agentqa_eval_recall", "recall {{strategy}} {{commit}}"),
            ("agentqa_eval_precision", "precision {{strategy}} {{commit}}"),
            ("agentqa_eval_triage_accuracy", "triage {{strategy}} {{commit}}"),
        ],
        unit="percentunit",
        w=12,
    )
    b.panel(
        "Hallucination rate",
        [("agentqa_eval_hallucination_rate", "{{strategy}} {{commit}}")],
        unit="percentunit",
        w=12,
    )
    return b


def perf() -> Board:
    b = Board("agentqa-perf", "AgentQA — load runs")
    b.row("Latency, throughput, errors")
    b.panel(
        "Latency by scenario (p95)",
        [('agentqa_perf_latency_seconds{quantile="p95"}', "{{build}} {{scenario}}")],
        unit="s",
        w=12,
    )
    b.panel("Throughput", [("agentqa_perf_rps", "{{build}} {{scenario}}")], unit="reqps", w=12)
    b.panel(
        "Error ratio", [("agentqa_perf_error_ratio", "{{build}} {{scenario}}")], unit="percentunit"
    )
    b.panel(
        "SLO verdicts (1 = met)", [("agentqa_perf_slo_verdict", "{{build}} {{slo}}")], kind="stat"
    )
    b.panel(
        "Run duration",
        [
            (
                "histogram_quantile(0.5, sum by (le) (rate(agentqa_run_duration_seconds_bucket[1h])))",
                "median run",
            )
        ],
        unit="s",
    )
    b.row("Target container (cAdvisor)")
    b.panel(
        "Target CPU",
        [('sum(rate(container_cpu_usage_seconds_total{name=~".*target.*"}[1m]))', "cores")],
        w=12,
    )
    b.panel(
        "Target memory",
        [('sum(container_memory_working_set_bytes{name=~".*target.*"})', "working set")],
        unit="bytes",
        w=12,
    )
    return b


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for board in (overview(), perf()):
        (OUT / f"{board.uid}.json").write_text(
            json.dumps(board.json(), indent=1) + "\n", encoding="utf-8"
        )
        print(f"wrote {board.uid}.json ({len(board.panels)} panels)")
