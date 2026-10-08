"""Load runner: Locust headless in a subprocess, ``iterations`` times, with warm-up discarded.

The target's own counters (/__metrics: per-route DB queries, DB time, bytes, CPU seconds, RSS)
are read before and after each iteration, so a slow request can be tied to its cause.
"""

from __future__ import annotations

import fnmatch
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

from agentqa import config
from agentqa.config import REPO_ROOT
from agentqa.guards import killswitch
from agentqa.obs import metrics, tracing
from agentqa.perf.models import IterationResult, PerfRun, StepStats, WorkloadSpec
from agentqa.target_config import TargetConfig

LOCUSTFILE = Path(__file__).with_name("locustfile.py")


class LoadGuardViolation(Exception):
    pass


def check_target_allowed(base_url: str, steps: list[list[float]]) -> None:
    """Load tests only against allowlisted sandbox targets, within user and duration caps."""
    g = config.guardrails()
    allowed = [
        t for t in g.targets.values() if t.sandbox and fnmatch.fnmatch(base_url, t.base_url_pattern)
    ]
    if not allowed:
        metrics.inc("agentqa_guardrail_events_total", guardrail="load", action="block")
        tracing.event(
            "guardrail.load",
            action="block",
            reason="target not an allowlisted sandbox",
            target=base_url,
        )
        raise LoadGuardViolation(
            f"{base_url} is not an allowlisted sandbox target (config/guardrails.yaml)"
        )
    duration = sum(s[0] for s in steps)
    if duration > g.load.max_duration_s:
        raise LoadGuardViolation(f"duration {duration}s exceeds max {g.load.max_duration_s}s")
    if max(s[1] for s in steps) > g.load.max_users:
        tracing.event(
            "guardrail.load",
            action="cap_users",
            requested=max(s[1] for s in steps),
            cap=g.load.max_users,
        )


def host_info() -> dict[str, object]:
    return {
        "cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "target_cpu_pinning": "1 core",
        "note": "shared build host; compare relatively (A/B), not absolutely",
    }


def target_metrics(base_url: str) -> dict[str, Any]:
    try:
        data: dict[str, Any] = httpx.get(f"{base_url}/__metrics", timeout=10).json()
        return data
    except (httpx.HTTPError, ValueError):
        return {}


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(q * (len(s) - 1))))
    return round(s[k], 2)


def aggregate(
    records: list[list[Any]], steps: list[list[float]], start: float, warmup_s: float
) -> list[StepStats]:
    """Bucket records into load steps by time; drop the first ``warmup_s`` of each step."""
    bounds = []
    t = start
    for duration, users, _rate in steps:
        bounds.append((t + warmup_s, t + duration, int(users)))
        t += duration
    buckets: dict[tuple[int, str], list[list[Any]]] = defaultdict(list)
    for r in records:
        for lo, hi, users in bounds:
            if lo <= r[0] < hi:
                buckets[(users, r[1])].append(r)
                buckets[(users, "all")].append(r)
                break
    durations = {users: hi - lo for lo, hi, users in bounds}
    out = []
    for (users, scenario), rows in sorted(buckets.items()):
        lat = [r[2] for r in rows]
        errs = [r for r in rows if not r[6]]
        by_status: dict[str, int] = defaultdict(int)
        for r in errs:
            by_status[str(r[4])] += 1
        out.append(
            StepStats(
                users=users,
                scenario=scenario,
                count=len(rows),
                rps=round(len(rows) / max(0.1, durations[users]), 2),
                p50_ms=_pct(lat, 0.5),
                p95_ms=_pct(lat, 0.95),
                p99_ms=_pct(lat, 0.99),
                error_rate=round(len(errs) / len(rows), 4),
                errors_by_status=dict(by_status),
                avg_bytes=round(statistics.fmean(r[3] for r in rows), 1),
            )
        )
    return out


def run_iteration(
    base_url: str,
    target: TargetConfig,
    workload: WorkloadSpec,
    steps: list[list[float]],
    out_dir: Path,
    iteration: int,
    warmup_s: float,
) -> IterationResult:
    g = config.guardrails().load
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / f"requests-{iteration}.jsonl"
    cfg = {
        "scenarios": [s.model_dump() for s in workload.scenarios],
        "think_time_s": workload.think_time_s,
        "headers": target.headers("admin") if "admin" in target.auth.roles else {},
        "customer_id": target.auth.roles["customer"].customer_id
        if "customer" in target.auth.roles
        else None,
        "webhook_secret": target.webhook.secret if target.webhook else "",
        "webhook_header": target.webhook.header if target.webhook else "X-Signature",
        "steps": steps,
        "out": str(records_path),
        "guard": {
            "max_users": g.max_users,
            "max_rps": g.max_rps,
            "abort_error_rate": g.abort_error_rate,
            "abort_host_cpu_pct": g.abort_host_cpu_pct,
            "abort_host_mem_pct": g.abort_host_mem_pct,
            "killswitch_file": str(killswitch.killswitch_path()),
        },
    }
    cfg_path = out_dir / f"perf-config-{iteration}.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    duration = int(sum(s[0] for s in steps)) + 2
    before = target_metrics(base_url)
    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(LOCUSTFILE),
        "--headless",
        "--host",
        base_url,
        "--run-time",
        f"{duration}s",
        "--only-summary",
        "--loglevel",
        "WARNING",
        "--stop-timeout",
        "2",
    ]
    env = {**os.environ, "AGENTQA_PERF_CONFIG": str(cfg_path), "PYTHONPATH": str(REPO_ROOT)}
    load_cpus = set(range(1, os.cpu_count() or 2)) or None  # target is pinned to core 0

    def pin() -> None:
        if load_cpus:
            os.sched_setaffinity(0, load_cpus)

    started = time.time()
    with tracing.span(
        "load iteration",
        "tool_call",
        **{"agentqa.iteration": iteration, "agentqa.target": base_url},
    ):
        proc = subprocess.run(  # noqa: S603 - locust with fixed arguments
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=duration + 60,
            preexec_fn=pin,
            check=False,
        )
    (out_dir / f"locust-{iteration}.log").write_text(
        proc.stdout[-5000:] + proc.stderr[-5000:], encoding="utf-8"
    )
    after = target_metrics(base_url)
    lines = records_path.read_text(encoding="utf-8").splitlines() if records_path.exists() else []
    header = (
        json.loads(lines[0])
        if lines
        else {"aborted": f"no records (locust exit {proc.returncode})"}
    )
    records = [json.loads(line) for line in lines[1:]]
    if header.get("aborted"):
        metrics.inc("agentqa_guardrail_events_total", guardrail="load", action="abort")
        tracing.event("guardrail.load", action="abort", reason=header["aborted"])
    first = min((r[0] for r in records), default=started)
    steps_stats = aggregate(records, steps, first, warmup_s)
    return IterationResult(
        iteration=iteration,
        steps=steps_stats,
        target_before=before,
        target_after=after,
        wall_s=round(time.time() - started, 2),
        aborted=header.get("aborted"),
    )


def run_perf(
    label: str,
    base_url: str,
    target: TargetConfig,
    workload: WorkloadSpec,
    test_type: str,
    out_dir: Path,
    iterations: int | None = None,
    steps: list[list[float]] | None = None,
) -> PerfRun:
    pd = config.perf_defaults()
    steps = steps or pd.shapes[test_type]
    check_target_allowed(base_url, steps)
    n = iterations or pd.iterations
    with tracing.span(f"perf {test_type} {label}", "stage", **{"agentqa.perf_label": label}):
        its = []
        for i in range(n):
            killswitch.check(f"perf iteration {i}")
            its.append(
                run_iteration(base_url, target, workload, steps, out_dir / label, i, pd.warmup_s)
            )
            if its[-1].aborted:
                break
    run = PerfRun(
        label=label, test_type=test_type, workload=workload, iterations=its, host=host_info()
    )
    _export(run)
    return run


def _export(run: PerfRun) -> None:
    """Publish the median of the last step per scenario as perf gauges (live on the dashboard)."""
    if not run.iterations:
        return
    last_users = max(s.users for s in run.iterations[0].steps) if run.iterations[0].steps else 0
    for it in run.iterations[-1:]:
        for s in it.steps:
            if s.users != last_users:
                continue
            for q, v in (("p50", s.p50_ms), ("p95", s.p95_ms), ("p99", s.p99_ms)):
                metrics.gauge(
                    "agentqa_perf_latency_seconds",
                    v / 1000,
                    scenario=s.scenario,
                    quantile=q,
                    build=run.label,
                )
            metrics.gauge("agentqa_perf_rps", s.rps, scenario=s.scenario, build=run.label)
            metrics.gauge(
                "agentqa_perf_error_ratio", s.error_rate, scenario=s.scenario, build=run.label
            )
