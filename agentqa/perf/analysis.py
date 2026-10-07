"""Deterministic performance analysis (no model involved).

- per-step medians across iterations (with min/max spread)
- knee point: Kneedle-style on the normalised throughput-vs-users curve, plus a latency knee
  (first step whose p95 is at least 2x the first step's)
- SLO verdicts at the step closest to the documented peak
- target-side signals from /__metrics deltas: DB queries per request, DB time share, bytes per
  request, CPU utilisation, memory growth
- A/B comparison with a noise band measured from clean-vs-clean repeats
- a compact evidence bundle per comparison for the perf-triage model
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from agentqa import config
from agentqa.perf.models import PerfRun, SLOs, WorkloadSpec

KEY_METRICS = ("p95_ms", "rps", "error_rate", "avg_bytes", "count")
MIN_SAMPLES = (
    50  # scenario cells with fewer requests are reported but never used to flag or to set noise
)


def summarize(run: PerfRun) -> dict[str, dict[int, dict[str, float]]]:
    """{scenario: {users: {metric: median, metric_min, metric_max}}} across iterations."""
    acc: dict[str, dict[int, dict[str, list[float]]]] = {}
    for it in run.iterations:
        for s in it.steps:
            row = acc.setdefault(s.scenario, {}).setdefault(
                s.users, {m: [] for m in (*KEY_METRICS, "p99_ms", "p50_ms")}
            )
            for m in row:
                row[m].append(float(getattr(s, m)))
    out: dict[str, dict[int, dict[str, float]]] = {}
    for scen, steps in acc.items():
        for users, row in steps.items():
            cell: dict[str, float] = {}
            for m, values in row.items():
                cell[m] = round(statistics.median(values), 4)
                cell[f"{m}_min"], cell[f"{m}_max"] = round(min(values), 4), round(max(values), 4)
            out.setdefault(scen, {})[users] = cell
    return out


def knee(users: list[int], rps: list[float], p95: list[float]) -> dict[str, int | None]:
    """Throughput knee: the step with the largest gap above the straight line between the first and
    last points of the normalised throughput curve. Latency knee: first step with p95 >= 2x the
    first step's p95."""
    out: dict[str, int | None] = {
        "throughput_knee_users": None,
        "latency_knee_users": None,
        "saturated_from_start": 0,
    }
    if users and rps and (max(rps) - min(rps)) / max(max(rps), 1e-9) < 0.15:
        # flat curve: already saturated at the first step; any "knee" inside it would be noise
        out["throughput_knee_users"], out["saturated_from_start"] = users[0], 1
    elif len(users) >= 3 and max(rps) > min(rps):
        x0, x1 = users[0], users[-1]
        y0, y1 = min(rps), max(rps)
        diffs = [
            ((r - y0) / (y1 - y0)) - ((u - x0) / (x1 - x0)) for u, r in zip(users, rps, strict=True)
        ]
        out["throughput_knee_users"] = users[max(range(len(diffs)), key=lambda i: diffs[i])]
    if users and p95 and p95[0] > 0:
        out["latency_knee_users"] = next(
            (u for u, p in zip(users, p95, strict=True) if p >= 2 * p95[0]), None
        )
    return out


def curve(
    summary: dict[str, dict[int, dict[str, float]]], scenario: str = "all"
) -> tuple[list[int], list[float], list[float]]:
    steps = summary.get(scenario, {})
    users = sorted(steps)
    return users, [steps[u]["rps"] for u in users], [steps[u]["p95_ms"] for u in users]


def target_signals(run: PerfRun) -> dict[str, float]:
    """Median across iterations of server-side signals from /__metrics deltas."""
    per_it: dict[str, list[float]] = {}
    for it in run.iterations:
        b, a = it.target_before, it.target_after
        if not a or not b:
            continue
        rb: dict[str, Any] = b.get("routes", {})
        ra: dict[str, Any] = a.get("routes", {})
        tot = {
            k: sum(float(ra.get(r, {}).get(k, 0)) - float(rb.get(r, {}).get(k, 0)) for r in ra)
            for k in ("requests", "db_queries", "db_ms", "total_ms", "response_bytes", "errors")
        }
        reqs = max(1.0, tot["requests"])
        sig = {
            "db_queries_per_request": tot["db_queries"] / reqs,
            "db_time_share": tot["db_ms"] / max(1.0, tot["total_ms"]),
            "bytes_per_request": tot["response_bytes"] / reqs,
            "server_ms_per_request": tot["total_ms"] / reqs,
            "server_5xx_ratio": tot["errors"] / reqs,
            "cpu_utilisation": (float(a.get("cpu_s", 0)) - float(b.get("cpu_s", 0)))
            / max(0.1, it.wall_s),
            "rss_growth_mb": float(a.get("rss_mb", 0)) - float(b.get("rss_mb", 0)),
            "rss_growth_mb_per_1k_requests": (float(a.get("rss_mb", 0)) - float(b.get("rss_mb", 0)))
            / reqs
            * 1000,
        }
        lo_b, lo_a = rb.get("GET /orders", {}), ra.get("GET /orders", {})
        if lo_a:
            n = max(1.0, float(lo_a.get("requests", 0)) - float(lo_b.get("requests", 0)))
            sig["list_db_queries_per_request"] = (
                float(lo_a.get("db_queries", 0)) - float(lo_b.get("db_queries", 0))
            ) / n
            sig["list_db_time_share"] = (
                float(lo_a.get("db_ms", 0)) - float(lo_b.get("db_ms", 0))
            ) / max(1.0, float(lo_a.get("total_ms", 0)) - float(lo_b.get("total_ms", 0)))
        for k, v in sig.items():
            per_it.setdefault(k, []).append(v)
    return {k: round(statistics.median(v), 4) for k, v in per_it.items()}


def slo_verdicts(run: PerfRun, workload: WorkloadSpec) -> list[dict[str, Any]]:
    """Each SLO checked at the step closest to the workload's peak users. Numbers quoted from the
    docs are 'documented'; missing ones fall back to config/perf_defaults.yaml as 'assumption'."""
    summary = summarize(run)
    slos = list(workload.slos)
    documented = bool(slos)
    if not documented:  # nothing quoted from the docs: defaults, labelled as assumptions
        d = config.perf_defaults().slos
        slos = [
            SLOs(
                p95_ms=d.get("p95_ms"),
                p99_ms=d.get("p99_ms"),
                error_rate=d.get("error_rate"),
                min_rps=d.get("min_rps"),
                scope="all",
            )
        ]
    out: list[dict[str, Any]] = []
    for slo in slos:
        scope = slo.scope if slo.scope in summary else "all"
        steps = summary.get(scope, {})
        if not steps:
            continue
        users = min(steps, key=lambda u: abs(u - workload.peak_users))
        cell = steps[users]
        for name, actual, limit, ok_if in (
            ("p95_ms", cell["p95_ms"], slo.p95_ms, "le"),
            ("p99_ms", cell["p99_ms"], slo.p99_ms, "le"),
            ("error_rate", cell["error_rate"], slo.error_rate, "le"),
            ("min_rps", cell["rps"], slo.min_rps, "ge"),
        ):
            if limit is None:
                continue
            source = "documented" if documented else "assumption"
            met = actual <= limit if ok_if == "le" else actual >= limit
            out.append(
                {
                    "slo": f"{scope}.{name}",
                    "limit": limit,
                    "actual": actual,
                    "users": users,
                    "met": met,
                    "source": source,
                }
            )
    return out


def noise_band(a: PerfRun, b: PerfRun) -> dict[str, float]:
    """Largest relative change between two runs of the same (clean) build, per metric, with a floor.
    A defect must exceed this to count as a regression."""
    sa, sb = summarize(a), summarize(b)
    worst: dict[str, float] = {"p95_ms": 0.0, "rps": 0.0}
    for scen in set(sa) & set(sb):
        for users in set(sa[scen]) & set(sb[scen]):
            if min(sa[scen][users]["count"], sb[scen][users]["count"]) < MIN_SAMPLES:
                continue  # too few requests for a stable percentile
            for m in worst:
                x, y = sa[scen][users][m], sb[scen][users][m]
                if x > 0 and y > 0:
                    worst[m] = max(worst[m], abs(math.log(y / x)))
    ta, tb = target_signals(a), target_signals(b)
    sig_noise = {k: abs(tb.get(k, 0) - ta.get(k, 0)) for k in ta}
    floors = {"p95_ms": math.log(1.25), "rps": math.log(1.15)}
    return {
        "p95_log_ratio": round(max(floors["p95_ms"], 1.5 * worst["p95_ms"]), 4),
        "rps_log_ratio": round(max(floors["rps"], 1.5 * worst["rps"]), 4),
        "error_rate_abs": 0.005,
        "rss_growth_mb_abs": round(max(15.0, 1.5 * sig_noise.get("rss_growth_mb", 0)), 2),
        "observed": {k: round(v, 4) for k, v in worst.items()},  # type: ignore[dict-item]
    }


def compare(baseline: PerfRun, candidate: PerfRun, band: dict[str, float]) -> dict[str, Any]:
    """A/B deltas per scenario at the highest common step, flagged against the noise band."""
    sb, sc = summarize(baseline), summarize(candidate)
    flagged: list[dict[str, Any]] = []
    for scen in sorted(set(sb) & set(sc)):
        common = sorted(set(sb[scen]) & set(sc[scen]))
        if not common:
            continue
        u = common[-1]
        b, c = sb[scen][u], sc[scen][u]
        if min(b["count"], c["count"]) < MIN_SAMPLES and c["avg_bytes"] <= 3 * max(
            1.0, b["avg_bytes"]
        ):
            continue
        p95_lr = math.log(c["p95_ms"] / b["p95_ms"]) if b["p95_ms"] > 0 and c["p95_ms"] > 0 else 0.0
        rps_lr = math.log(c["rps"] / b["rps"]) if b["rps"] > 0 and c["rps"] > 0 else 0.0
        err_d = c["error_rate"] - b["error_rate"]
        bytes_ratio = c["avg_bytes"] / b["avg_bytes"] if b["avg_bytes"] > 0 else 1.0
        reasons = []
        if p95_lr > band["p95_log_ratio"]:
            reasons.append("p95")
        if rps_lr < -band["rps_log_ratio"]:
            reasons.append("throughput")
        if err_d > band["error_rate_abs"]:
            reasons.append("errors")
        if bytes_ratio > 3:
            reasons.append("payload")
        flagged.append(
            {
                "scenario": scen,
                "users": u,
                "p95_ratio": round(math.exp(p95_lr), 3),
                "rps_ratio": round(math.exp(rps_lr), 3),
                "error_rate_delta": round(err_d, 4),
                "bytes_ratio": round(bytes_ratio, 2),
                "regressed": bool(reasons),
                "reasons": reasons,
            }
        )
    tb, tc = target_signals(baseline), target_signals(candidate)
    rss_delta = tc.get("rss_growth_mb", 0) - tb.get("rss_growth_mb", 0)
    server: list[str] = []  # server-side signals (span/query counts, DB time) regress too
    if tc.get("list_db_queries_per_request", 0) > 2 * max(
        1.0, tb.get("list_db_queries_per_request", 0)
    ):
        server.append("db_queries_per_request")
    if tc.get("list_db_time_share", 0) > max(0.2, 2 * tb.get("list_db_time_share", 0)):
        server.append("db_time_share")
    if rss_delta > band["rss_growth_mb_abs"]:
        server.append("memory_growth")
    regressed = any(f["regressed"] for f in flagged) or bool(server)
    return {
        "scenarios": flagged,
        "regressed": regressed,
        "server_signals": server,
        "rss_growth_delta_mb": round(rss_delta, 2),
        "signals_baseline": tb,
        "signals_candidate": tc,
    }


def evidence_bundle(
    baseline: PerfRun, candidate: PerfRun, comparison: dict[str, Any]
) -> dict[str, Any]:
    """The compact, numbers-only bundle the perf-triage model sees (never raw logs)."""
    sb, sc = summarize(baseline), summarize(candidate)
    ub, rb, pb = curve(sb)
    uc, rc, pc = curve(sc)
    top = sorted(comparison["scenarios"], key=lambda f: (-len(f["reasons"]), -f["p95_ratio"]))[:4]
    errors_by_status: dict[str, int] = {}
    for it in candidate.iterations:
        for s in it.steps:
            if s.scenario == "all":
                for k, v in s.errors_by_status.items():
                    errors_by_status[k] = errors_by_status.get(k, 0) + v
    tb, tc = comparison["signals_baseline"], comparison["signals_candidate"]
    return {
        "load_steps_users": uc,
        "throughput_rps": {"baseline": rb, "candidate": rc},
        "p95_ms": {"baseline": pb, "candidate": pc},
        "error_rate_by_step": {
            "candidate": [sc["all"][u]["error_rate"] for u in uc] if "all" in sc else []
        },
        "errors_by_status": errors_by_status,
        "knee": {"baseline": knee(ub, rb, pb), "candidate": knee(uc, rc, pc)},
        "most_regressed_scenarios": top,
        "db_queries_per_request": {
            "baseline": tb.get("list_db_queries_per_request"),
            "candidate": tc.get("list_db_queries_per_request"),
        },
        "db_time_share": {
            "baseline": tb.get("list_db_time_share"),
            "candidate": tc.get("list_db_time_share"),
        },
        "cpu_utilisation": {
            "baseline": tb.get("cpu_utilisation"),
            "candidate": tc.get("cpu_utilisation"),
        },
        "rss_growth_mb": {
            "baseline": tb.get("rss_growth_mb"),
            "candidate": tc.get("rss_growth_mb"),
        },
        "bytes_per_request": {
            "baseline": tb.get("bytes_per_request"),
            "candidate": tc.get("bytes_per_request"),
        },
    }


def rule_diagnosis(e: dict[str, Any]) -> tuple[str, list[str]]:
    """Deterministic first guess, shown in the report next to the model's diagnosis so a reviewer
    can see when they disagree. The model receives only the evidence, not this guess.
    Returns (class, evidence keys)."""
    q, share = e["db_queries_per_request"], e["db_time_share"]
    cpu, rss = e["cpu_utilisation"], e["rss_growth_mb"]
    rps_b, rps_c = e["throughput_rps"]["baseline"], e["throughput_rps"]["candidate"]
    if (q["candidate"] or 0) > 3 * max(1.0, q["baseline"] or 1.0):
        return "n_plus_one", ["db_queries_per_request"]
    if (
        any(str(k).startswith("503") for k in e["errors_by_status"])
        and max(e["error_rate_by_step"]["candidate"] or [0]) > 0.02
    ):
        return "pool_exhaustion", ["errors_by_status", "error_rate_by_step"]
    if any(s["bytes_ratio"] > 10 for s in e["most_regressed_scenarios"]):
        return "unbounded_payload", ["most_regressed_scenarios"]
    if (share["candidate"] or 0) > 0.25 and (share["candidate"] or 0) > 2.5 * max(
        0.01, share["baseline"] or 0
    ):
        return "missing_index", ["db_time_share", "p95_ms"]
    if (rss["candidate"] or 0) > 1.8 * max(10.0, rss["baseline"] or 0):
        return "memory_leak", ["rss_growth_mb"]
    if (
        rps_b
        and rps_c
        and max(rps_c) < 0.75 * max(rps_b)
        and (cpu["candidate"] or 0) < 0.75 * (cpu["baseline"] or 1)
    ):
        return "blocking_handler", ["throughput_rps", "cpu_utilisation"]
    return "none", []
