from __future__ import annotations

import math

import pytest

from agentqa.perf import analysis
from agentqa.perf.models import IterationResult, PerfRun, Scenario, SLOs, StepStats, WorkloadSpec
from agentqa.perf.runner import LoadGuardViolation, aggregate, check_target_allowed
from agentqa.sim.perf_sim import perf_triage  # noqa: F401  (registers responder)

WL = WorkloadSpec(
    scenarios=[
        Scenario(name="list_orders", method="GET", path="/orders", weight=1, expected_status=[200])
    ],
    think_time_s=0,
    data_setup_orders=0,
    ramp_profile="steps",
    slos=[SLOs(p95_ms=300, error_rate=0.01, scope="list_orders")],
    slo_source=["doc:x"],
    peak_users=20,
)


def run(
    label: str,
    rps: list[float],
    p95: list[float],
    err: float = 0.0,
    q: float = 3.0,
    share: float = 0.05,
    rss: float = 50.0,
    bytes_: float = 500.0,
    cpu: float = 20.0,
) -> PerfRun:
    users = [5, 15, 30, 50]
    steps = []
    for u, r, p in zip(users, rps, p95, strict=True):
        for scen in ("all", "list_orders"):
            steps.append(
                StepStats(
                    users=u,
                    scenario=scen,
                    count=500,
                    rps=r,
                    p50_ms=p / 2,
                    p95_ms=p,
                    p99_ms=p * 1.5,
                    error_rate=err,
                    errors_by_status={"503": int(err * 500)} if err else {},
                    avg_bytes=bytes_,
                )
            )
    before = {
        "rss_mb": 50.0,
        "cpu_s": 0.0,
        "routes": {
            "GET /orders": {
                "requests": 0,
                "db_queries": 0,
                "db_ms": 0,
                "total_ms": 0,
                "response_bytes": 0,
                "errors": 0,
            }
        },
    }
    after = {
        "rss_mb": 50.0 + rss,
        "cpu_s": cpu,
        "routes": {
            "GET /orders": {
                "requests": 1000,
                "db_queries": 1000 * q,
                "db_ms": 1000 * share * 10,
                "total_ms": 1000 * 10,
                "response_bytes": 1000 * bytes_,
                "errors": 0,
            }
        },
    }
    its = [
        IterationResult(
            iteration=i, steps=steps, target_before=before, target_after=after, wall_s=25.0
        )
        for i in range(3)
    ]
    return PerfRun(label=label, test_type="stress", workload=WL, iterations=its)


CLEAN = run("clean", [400, 420, 425, 430], [50, 70, 100, 140])
CLEAN2 = run("clean2", [410, 415, 430, 425], [52, 68, 105, 145])


def test_aggregate_buckets_and_warmup() -> None:
    recs = [[100.0 + t / 10, "a", 10.0 + t, 100, 200, 5, True] for t in range(100)]
    recs.append([107.0, "a", 999.0, 10, 503, 5, False])
    stats = aggregate(recs, [[5, 5, 5], [5, 10, 5]], start=100.0, warmup_s=1.0)
    first = next(s for s in stats if s.users == 5 and s.scenario == "a")
    assert first.count == 40  # 100.0-101.0 discarded as warm-up; 107.0 belongs to step 2
    second = next(s for s in stats if s.users == 10 and s.scenario == "all")
    assert second.error_rate > 0 and second.errors_by_status == {"503": 1}


def test_knee_and_saturation() -> None:
    k = analysis.knee([5, 15, 30, 50], [100, 280, 300, 305], [20, 25, 60, 120])
    assert k["throughput_knee_users"] == 15 and k["latency_knee_users"] == 30
    flat = analysis.knee([5, 15, 30, 50], [400, 410, 405, 402], [50, 70, 100, 140])
    assert flat["saturated_from_start"] == 1 and flat["throughput_knee_users"] == 5


def test_noise_band_and_no_false_alarm() -> None:
    band = analysis.noise_band(CLEAN, CLEAN2)
    assert band["p95_log_ratio"] >= round(math.log(1.25), 4)
    assert not analysis.compare(CLEAN, CLEAN2, band)["regressed"]


@pytest.mark.parametrize(
    "cand,expected",
    [
        (run("P01", [380, 400, 410, 410], [55, 75, 110, 150], q=80), "n_plus_one"),
        (run("P02", [100, 90, 95, 100], [110, 300, 500, 700], share=0.4), "missing_index"),
        (run("P03", [300, 300, 300, 290], [60, 100, 160, 260], bytes_=60000), "unbounded_payload"),
        (run("P04", [220, 190, 200, 190], [66, 180, 240, 390], cpu=10), "blocking_handler"),
        (run("P05", [400, 450, 440, 450], [50, 67, 100, 150], err=0.1), "pool_exhaustion"),
        (run("P06", [380, 340, 330, 290], [54, 77, 130, 218], rss=240), "memory_leak"),
    ],
)
def test_ab_flags_and_rule_diagnosis(cand: PerfRun, expected: str) -> None:
    band = analysis.noise_band(CLEAN, CLEAN2)
    cmp = analysis.compare(CLEAN, cand, band)
    assert cmp["regressed"]
    bundle = analysis.evidence_bundle(CLEAN, cand, cmp)
    assert analysis.rule_diagnosis(bundle)[0] == expected


def test_slo_verdicts_documented_and_assumed() -> None:
    v = analysis.slo_verdicts(run("x", [400] * 4, [100, 200, 400, 800]), WL)
    p95 = next(x for x in v if x["slo"] == "list_orders.p95_ms")
    assert p95["users"] == 15 and p95["met"] and p95["source"] == "documented"
    assumed = analysis.slo_verdicts(CLEAN, WL.model_copy(update={"slos": []}))
    assert assumed and all(x["source"] == "assumption" for x in assumed)


def test_load_guard_allowlist() -> None:
    check_target_allowed("http://127.0.0.1:8123", [[10, 5, 5]])
    with pytest.raises(LoadGuardViolation):
        check_target_allowed("https://api.production.example.com", [[10, 5, 5]])
    with pytest.raises(LoadGuardViolation):
        check_target_allowed("http://127.0.0.1:8123", [[100000, 5, 5]])
