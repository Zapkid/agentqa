"""One eval case = one pipeline run (all-bugs build as target, clean build as reference) plus the
differential ground truth for the suite it produced, scored into a flat metrics dict.

Each case runs cold: a fresh SQLite store (no lessons, no learned stats, no incremental
artifacts), an ephemeral vector store and the LLM disk cache off, so cost comparisons are never
flattered by earlier runs. Cache and memory effects are measured by dedicated cases.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from agentqa.agents.triage import Triage
from agentqa.evals.groundtruth import BUG_CATEGORY, BUG_IDS, GroundTruth, differential
from agentqa.ingest.vectorstore import VectorStore
from agentqa.llm.cache import DiskCache
from agentqa.llm.pricing import CostLedger
from agentqa.llm.router import ModelRouter
from agentqa.orchestrator.pipeline import Pipeline, RunConfig, RunOutput
from agentqa.store import Store
from agentqa.target_config import TargetConfig, load_target
from target_api.launcher import TargetServer

ROOT = Path(__file__).resolve().parent.parent.parent


def run_case(
    *,
    profile: str,
    strategy: str,
    seed: int,
    case_dir: Path,
    mechanism_overrides: dict[str, bool] | None = None,
    store: Store | None = None,
    cache_mode: str = "off",
    target: TargetConfig | None = None,
    use_lessons: bool = True,
    run_id: str | None = None,
) -> dict[str, Any]:
    target = target or load_target()
    case_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AGENTQA_SIM_SEED"] = str(seed)
    if store is None:
        (case_dir / "agentqa.db").unlink(missing_ok=True)  # every case starts cold
        store = Store(case_dir / "agentqa.db")
    t0 = time.monotonic()
    with TargetServer(bugs=BUG_IDS) as tgt, TargetServer() as ref:
        out = Pipeline(
            RunConfig(
                spec=ROOT / "target_api/openapi.json",
                docs=ROOT / "target_api/docs",
                target=target,
                base_url=tgt.base_url,
                reference_url=ref.base_url,
                profile=profile,
                strategy=strategy,
                mechanism_overrides=mechanism_overrides or {},
                run_id=run_id or f"{strategy}-s{seed}",
                out_root=case_dir,
                cache_mode=cache_mode,
                server_log=tgt.log_lines,
                store=store,
                vector_store=VectorStore(),
                use_lessons=use_lessons,
            )
        ).run()
    pipeline_s = time.monotonic() - t0
    gt = differential(out.tests, target, case_dir / "differential", all_bugs_run=out.run_result)
    metrics = score(out, gt, profile)
    metrics.update(
        {
            "profile": profile,
            "strategy": strategy,
            "seed": seed,
            "pipeline_wall_s": round(pipeline_s, 1),
        }
    )
    (case_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=1, default=str), encoding="utf-8"
    )
    return metrics


def _false_positives_on_clean(out: RunOutput, gt: GroundTruth, profile: str) -> tuple[int, int]:
    """Triage the suite's failures on the clean build with the same models (separate ledger, not
    charged to the strategy) and count product_bug verdicts: every one of them is a false alarm."""
    if gt.clean_run is None or out.bundle is None:
        return 0, 0
    failing = [r for r in gt.clean_run.results if r.outcome in ("failed", "error")]
    if not failing:
        return 0, 0
    router = ModelRouter(profile, cache=DiskCache(mode="off"), ledger=CostLedger())
    intents = {vt.intent.id: vt.intent for vt in out.tests}
    findings = Triage(out.bundle).triage(gt.clean_run, intents, router.for_tier)
    return sum(f.classification == "product_bug" for f in findings), len(findings)


def _llm_latencies(run_dir: Path) -> list[float]:
    path = run_dir / "spans.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        a = json.loads(line)["attributes"]
        if a.get("agentqa.span_kind") == "llm_call" and a.get("agentqa.cache_status") != "hit":
            out.append(float(a.get("agentqa.latency_s", 0)))
    return out


def score(out: RunOutput, gt: GroundTruth, profile: str) -> dict[str, Any]:
    r = out.report
    detected = gt.detected()
    by_cat: dict[str, list[int]] = {}
    for bug in BUG_IDS:
        by_cat.setdefault(BUG_CATEGORY[bug], []).append(int(bug in detected))
    # triage accuracy: every failing test in the all-bugs run, predicted vs differential truth
    predicted: dict[str, str] = {}
    for f in r.findings:
        for t in f.test_names:
            predicted[t] = f.classification
    confusion: Counter[str] = Counter()
    correct = considered = 0
    for res in r.results:
        if res.outcome not in ("failed", "error", "blocked"):
            continue
        truth = gt.label(res.test_name)
        truth = "product_bug" if truth == "interaction" else truth
        if truth == "pass":
            truth = "flaky"  # failed in the run but never reproduced: behaves like a flaky test
        pred = predicted.get(res.test_name, "untriaged")
        confusion[f"{truth}->{pred}"] += 1
        considered += 1
        correct += int(pred == truth)
    product = [f for f in r.findings if f.classification == "product_bug"]
    true_product = [
        f
        for f in product
        if any(gt.label(t) in ("product_bug", "interaction") for t in f.test_names)
    ]
    fp_clean, clean_findings = _false_positives_on_clean(out, gt, profile)
    lat = _llm_latencies(out.run_dir)
    cost = r.cost
    bugs_found = len(detected)
    by_agent = cost.get("by_agent", {})
    tests_total = len(out.tests)
    return {
        "bugs_detected": sorted(detected),
        "recall": round(bugs_found / len(BUG_IDS), 4),
        "recall_by_category": {c: round(sum(v) / len(v), 3) for c, v in sorted(by_cat.items())},
        "precision": round(len(true_product) / len(product), 4) if product else None,
        "product_findings": len(product),
        "false_positive_findings_on_clean": fp_clean,
        "findings_on_clean": clean_findings,
        "triage_accuracy": round(correct / considered, 4) if considered else None,
        "triage_confusion": dict(confusion),
        "test_validity_rate": round(gt.validity_rate(), 4),
        "hallucination_rate": round(float(r.hallucination.get("rate", 0.0)), 4),
        "flakiness_rate": round(len(gt.flaky_tests) / tests_total, 4) if tests_total else 0.0,
        "tests": tests_total,
        "tests_by_tier": dict(Counter(vt.tier for vt in out.tests)),
        "tokens_total": cost.get("total_tokens", 0),
        "tokens_by_stage": {
            a: int(v["input_tokens"] + v["output_tokens"]) for a, v in by_agent.items()
        },
        "cached_input_tokens": cost.get("cached_input_tokens", 0),
        "llm_calls": cost.get("calls", 0),
        "cost_usd_actual": cost.get("cost_usd_actual", 0.0),
        "cost_usd_list_equivalent": cost.get("cost_usd_list_equivalent", 0.0),
        "cost_per_bug_list_equivalent": round(
            cost.get("cost_usd_list_equivalent", 0.0) / bugs_found, 6
        )
        if bugs_found
        else None,
        "llm_latency_p95_s": round(statistics.quantiles(lat, n=20)[-1], 4)
        if len(lat) >= 2
        else (lat[0] if lat else 0.0),
        "escalation_rate": r.delegation.get("escalation_rate", 0.0),
        "escalations": r.delegation.get("escalations", 0),
        "uncovered": len(r.uncovered),
        "uncovered_by_risk": dict(Counter(str(u.risk) for u in r.uncovered)),
        "judge": r.judge.model_dump() if r.judge else None,
        "aborted": r.aborted,
        "run_wall_s": out.stats.get("elapsed_s"),
        "run_dir": str(out.run_dir),
    }


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and spread (min, max, stdev) of every numeric metric across seeds."""
    keys = [
        k
        for k, v in cases[0].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "seed"
    ]
    out: dict[str, Any] = {"n": len(cases)}
    for k in keys:
        vals = [c[k] for c in cases if isinstance(c.get(k), (int, float))]
        if not vals:
            continue
        out[k] = {
            "mean": round(statistics.fmean(vals), 6),
            "min": round(min(vals), 6),
            "max": round(max(vals), 6),
            "stdev": round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0,
        }
    cats: dict[str, list[float]] = {}
    for c in cases:
        for cat, v in c.get("recall_by_category", {}).items():
            cats.setdefault(cat, []).append(v)
    out["recall_by_category"] = {k: round(statistics.fmean(v), 3) for k, v in cats.items()}
    detected = Counter(b for c in cases for b in c.get("bugs_detected", []))
    out["bug_detection_frequency"] = {b: f"{detected.get(b, 0)}/{len(cases)}" for b in BUG_IDS}
    return out
