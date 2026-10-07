"""F11 performance flow: workload model -> load runs -> deterministic analysis -> perf triage.

``perf_ab`` is the relative A/B check used in CI and in the benchmark: the same WorkloadSpec runs
against a baseline build and a candidate build back to back on the same host; the candidate is
flagged only when it degrades beyond the noise band measured from clean-vs-clean repeats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from agentqa.config import agentqa_home
from agentqa.ingest.ingestor import ingest
from agentqa.ingest.vectorstore import VectorStore
from agentqa.llm.cache import DiskCache
from agentqa.llm.router import ModelRouter
from agentqa.obs import metrics, tracing
from agentqa.perf import analysis
from agentqa.perf.charts import load_charts
from agentqa.perf.models import PerfRun, PerfVerdict, WorkloadSpec
from agentqa.perf.runner import run_perf
from agentqa.perf.triage import diagnose
from agentqa.perf.workload import model_workload
from agentqa.target_config import TargetConfig
from target_api.launcher import TargetServer


@dataclass
class ABResult:
    baseline: PerfRun
    candidate: PerfRun
    band: dict[str, Any]
    comparison: dict[str, Any]
    evidence: dict[str, Any] | None = None
    verdict: PerfVerdict | None = None
    rule_class: str | None = None
    triage_path: list[str] = field(default_factory=list)
    chart: Path | None = None
    slo: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.label,
            "candidate": self.candidate.label,
            "regressed": self.comparison["regressed"],
            "band": self.band,
            "diagnosis": self.verdict.bottleneck if self.verdict else None,
            "rule_diagnosis": self.rule_class,
            "confidence": self.verdict.confidence if self.verdict else None,
            "fix": self.verdict.fix if self.verdict else None,
            "triage_path": self.triage_path,
            "flagged": [s for s in self.comparison["scenarios"] if s["regressed"]],
            "server_signals": self.comparison.get("server_signals", []),
            "slo": self.slo,
            "chart": str(self.chart) if self.chart else None,
        }


class PerfSession:
    """Shared context: one workload model per session (one small-model call), reused for every build."""

    def __init__(
        self,
        target: TargetConfig,
        spec: str | Path,
        docs: str | Path | None,
        profile: str = "simulated",
        out_dir: Path | None = None,
        cache_mode: str | None = None,
    ) -> None:
        self.target = target
        self.out_dir = out_dir or agentqa_home() / "perf"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.store = VectorStore()
        self.bundle, _ = ingest(spec, docs, self.store)
        self.router = ModelRouter(
            profile, cache=DiskCache(mode=cast(Any, cache_mode)) if cache_mode else DiskCache()
        )
        self._workload: WorkloadSpec | None = None
        self.workload_path: list[str] = []

    @property
    def clients(self) -> dict[str, Any]:
        return {"T1": self.router.for_tier("T1"), "T2": self.router.for_tier("T2")}

    def workload(self) -> WorkloadSpec:
        if self._workload is None:
            self._workload, self.workload_path = model_workload(
                self.bundle, self.store, self.clients
            )
            (self.out_dir / "workload.json").write_text(
                self._workload.model_dump_json(indent=1), encoding="utf-8"
            )
        return self._workload

    def run_build(
        self,
        label: str,
        perf_bugs: list[str],
        test_type: str,
        iterations: int | None,
        seed_orders: int | None,
    ) -> PerfRun:
        wl = self.workload()
        orders = wl.data_setup_orders if seed_orders is None else seed_orders
        with TargetServer(perf_bugs=perf_bugs, cpus={0}) as srv:
            if orders:
                srv.seed(orders)
            return run_perf(
                label, srv.base_url, self.target, wl, test_type, self.out_dir, iterations
            )

    def ab(
        self, baseline: PerfRun, candidate: PerfRun, band: dict[str, Any], task_id: str
    ) -> ABResult:
        with tracing.span(f"perf ab {candidate.label}", "stage"):
            comparison = analysis.compare(baseline, candidate, band)
            res = ABResult(
                baseline,
                candidate,
                band,
                comparison,
                slo=analysis.slo_verdicts(candidate, self.workload()),
            )
            for v in res.slo:
                metrics.gauge(
                    "agentqa_perf_slo_verdict",
                    1.0 if v["met"] else 0.0,
                    slo=v["slo"],
                    build=candidate.label,
                )
            if comparison["regressed"]:
                res.evidence = analysis.evidence_bundle(baseline, candidate, comparison)
                res.rule_class, _ = analysis.rule_diagnosis(res.evidence)
                res.verdict, res.triage_path = diagnose(res.evidence, self.clients, task_id)
            res.chart = load_charts(
                {"baseline": baseline, "candidate": candidate},
                self.out_dir / f"chart-{baseline.label}-vs-{candidate.label}.png",
                f"{candidate.test_type}: {baseline.label} vs {candidate.label}",
            )
        (self.out_dir / f"ab-{candidate.label}.json").write_text(
            json.dumps(res.summary(), indent=1, default=str)
        )
        return res


def perf_report(
    results: list[ABResult], workload: WorkloadSpec, workload_path: list[str], out: Path
) -> Path:
    lines = [
        "# Performance report",
        "",
        "Relative A/B runs on one host (target pinned to one CPU core); absolute numbers describe this",
        "host only. A build is flagged when it degrades beyond the measured clean-vs-clean noise band.",
        "",
        f"Workload: {len(workload.scenarios)} scenarios, think time {workload.think_time_s}s, "
        f"{workload.data_setup_orders} seeded orders; modelled via {' -> '.join(workload_path)}.",
        f"SLO sources: {', '.join(workload.slo_source) or 'none (defaults used, labelled assumptions)'}",
        "",
    ]
    for r in results:
        s = r.summary()
        lines += [
            f"## {s['candidate']} vs {s['baseline']}",
            "",
            f"- Regressed beyond noise: **{s['regressed']}** (server-side signals: {', '.join(s['server_signals']) or 'none'})",
            f"- Diagnosis: **{s['diagnosis']}** (confidence {s['confidence']}; rule-based check: {s['rule_diagnosis']}; path {s['triage_path']})",
            f"- Suggested fix: {s['fix']}" if s["fix"] else "- Suggested fix: none needed",
        ]
        for f in s["flagged"]:
            lines.append(
                f"  - {f['scenario']} @ {f['users']} users: p95 x{f['p95_ratio']}, throughput x{f['rps_ratio']}, "
                f"errors {f['error_rate_delta']:+.2%}, bytes x{f['bytes_ratio']} ({', '.join(f['reasons'])})"
            )
        for v in s["slo"]:
            lines.append(
                f"  - SLO {v['slo']} ({v['source']}): limit {v['limit']}, actual {v['actual']} at {v['users']} users "
                f"-> {'met' if v['met'] else 'VIOLATED'}"
            )
        if r.chart:
            lines += ["", f"![{s['candidate']}]({r.chart.name})", ""]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
