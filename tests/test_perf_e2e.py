"""Real load against local builds: one defect A/B with a measured noise band (slow)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentqa.perf import analysis
from agentqa.perf.pipeline import PerfSession
from agentqa.target_config import load_target

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parent.parent
STEPS = [[5, 5, 50], [5, 20, 50]]


def test_ab_detects_unbounded_payload(tmp_path: Path) -> None:
    s = PerfSession(
        load_target(),
        ROOT / "target_api/openapi.json",
        ROOT / "target_api/docs",
        out_dir=tmp_path,
        cache_mode="off",
    )
    wl = s.workload()
    assert {sc.name for sc in wl.scenarios} >= {"list_orders", "payment_webhook"}
    assert wl.slo_source and wl.slos and wl.slos[0].p95_ms == 300

    def build(label: str, bugs: list[str]):  # type: ignore[no-untyped-def]
        from agentqa.perf.runner import run_perf
        from target_api.launcher import TargetServer

        with TargetServer(perf_bugs=bugs, cpus={0}) as srv:
            srv.seed(5000)
            return run_perf(
                label, srv.base_url, s.target, wl, "stress", tmp_path, iterations=2, steps=STEPS
            )

    clean, clean2, cand = build("clean", []), build("clean2", []), build("P03", ["P03"])
    band = analysis.noise_band(clean, clean2)
    assert not analysis.compare(clean, clean2, band)["regressed"]
    res = s.ab(clean, cand, band, "test-p03")
    assert res.comparison["regressed"] and res.rule_class == "unbounded_payload"
    assert res.chart and res.chart.exists()
