"""`make demo`: one command for the 5-minute demo (see docs/DEMO.md).

1. If Docker is available, start the observability stack (Phoenix, Collector, Prometheus, Grafana)
   and export traces/metrics to it; otherwise run fully local (spans saved per run).
2. Run the functional pipeline against the bundled Orders API with every seeded bug on, using a
   clean build as the reference (strategy S3, profile from AGENTQA_PROFILE, default simulated).
3. Run a short performance A/B (clean vs P01 N+1) with diagnosis and charts.
4. Print where to look: report, executive summary, UI, Phoenix, Grafana.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def docker_stack() -> bool:
    if shutil.which("docker") is None:
        return False
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False
    print("== starting observability stack (docker compose)")
    subprocess.run(
        ["docker", "compose", "-f", str(ROOT / "deploy/docker-compose.yml"), "up", "-d", "--build"],
        check=True,
    )
    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    os.environ["AGENTQA_TRACING"] = "1"
    time.sleep(5)
    return True


def main() -> None:
    stack = docker_stack()
    if not stack:
        print("== Docker not available: running fully local (spans are saved with each run)")
    profile = os.environ.get("AGENTQA_PROFILE", "simulated")
    from agentqa.cli import execute_run

    print(
        f"== functional run: profile {profile}, strategy S3, target = all seeded bugs, reference = clean build"
    )
    out = execute_run(
        spec=None,
        docs=None,
        profile=profile,
        strategy="S3",
        target_config=ROOT / "target_api/agentqa_target.yaml",
        base_url=None,
        reference_url=None,
        local_target="all",
        local_reference=True,
        cache_mode=None,
        run_id=None,
        max_tokens=None,
    )
    r = out.report
    print(
        f"   outcomes {r.outcome_counts()} | findings {len(r.findings)} ({len(r.product_bugs())} product bugs, "
        f"{len(r.blocking())} blocking) | tokens {r.cost['total_tokens']} | list ${r.cost['cost_usd_list_equivalent']:.4f}"
    )
    print(
        f"   tests by tier: {dict(__import__('collections').Counter(r.tiers.values()))} | "
        f"escalations {r.delegation['escalations']} | quarantined doc chunks {r.quarantined_chunks}"
    )

    print("== performance A/B: clean vs P01 (N+1), 2 iterations, short stepped load")
    from agentqa.perf import analysis
    from agentqa.perf.pipeline import PerfSession, perf_report
    from agentqa.target_config import load_target

    target = load_target()
    s = PerfSession(target, target.spec_path, target.docs_path, profile=profile)
    steps = [[5, 5, 50], [5, 20, 50], [5, 40, 50]]

    def build(label: str, bugs: list[str]):  # type: ignore[no-untyped-def]
        from agentqa.perf.runner import run_perf
        from target_api.launcher import TargetServer

        with TargetServer(perf_bugs=bugs, cpus={0}) as srv:
            srv.seed(20_000)
            return run_perf(
                label,
                srv.base_url,
                target,
                s.workload(),
                "stress",
                s.out_dir,
                iterations=2,
                steps=steps,
            )

    base, repeat, cand = build("clean", []), build("clean-repeat", []), build("P01", ["P01"])
    res = s.ab(base, cand, analysis.noise_band(base, repeat), "demo-P01")
    perf_path = perf_report([res], s.workload(), s.workload_path, s.out_dir / "perf_report-demo.md")
    summ = res.summary()
    print(
        f"   regressed {summ['regressed']} ({summ['server_signals']}) | diagnosis {summ['diagnosis']} | fix: {summ['fix']}"
    )

    print("\n== where to look")
    print(f"   executive summary : {out.paths['summary_html']}")
    print(f"   technical report  : {out.paths['report_html']}")
    print(f"   perf report/chart : {perf_path}  {res.chart}")
    print(f"   trace id          : {r.trace_id} (spans in {out.run_dir / 'spans.jsonl'})")
    print("   web UI            : uv run agentqa serve  ->  http://127.0.0.1:8088")
    if stack:
        print("   Phoenix           : http://localhost:6006   Grafana: http://localhost:3000")


if __name__ == "__main__":
    main()
