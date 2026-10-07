"""F10 command line: ``agentqa run | report | replay | runs | killswitch | eval | perf | serve | mcp``."""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated, Any

import typer

from agentqa.config import REPO_ROOT, agentqa_home

app = typer.Typer(
    add_completion=False, help="AgentQA: autonomous, observable, guarded API testing."
)
DEFAULT_TARGET = REPO_ROOT / "target_api" / "agentqa_target.yaml"


def _bugs(value: str | None) -> list[str]:
    if not value or value == "clean":
        return []
    if value == "all":
        return [f"B{i:02d}" for i in range(1, 13)]
    return [b.strip().upper() for b in value.split(",") if b.strip()]


def execute_run(
    *,
    spec: str | None,
    docs: str | None,
    profile: str,
    strategy: str,
    target_config: Path,
    base_url: str | None,
    reference_url: str | None,
    local_target: str | None,
    local_reference: bool,
    cache_mode: str | None,
    run_id: str | None,
    max_tokens: int | None,
    use_lessons: bool = True,
    out_root: Path | None = None,
) -> Any:
    """Shared by the CLI, the MCP server and the API."""
    from agentqa.obs.logging import configure_logging
    from agentqa.orchestrator.pipeline import Pipeline, RunConfig
    from agentqa.target_config import load_target
    from target_api.launcher import TargetServer

    configure_logging(os.environ.get("AGENTQA_LOG_LEVEL", "WARNING"))
    target = load_target(target_config)
    with ExitStack() as stack:
        server_log = None
        if local_target is not None:
            srv = stack.enter_context(TargetServer(bugs=_bugs(local_target)))
            base_url, server_log = srv.base_url, srv.log_lines
        if local_reference:
            reference_url = stack.enter_context(TargetServer()).base_url
        if not base_url:
            raise typer.BadParameter("give --base-url or --local-target")
        budget: dict[str, float] = {"max_tokens": max_tokens} if max_tokens else {}
        cfg = RunConfig(
            spec=spec or target.spec_path,
            docs=docs if docs is not None else target.docs_path,
            target=target,
            base_url=base_url,
            reference_url=reference_url,
            profile=profile,
            strategy=strategy,
            run_id=run_id,
            cache_mode=cache_mode,
            server_log=server_log,
            budget=budget,
            use_lessons=use_lessons,
            out_root=out_root,
        )
        out = Pipeline(cfg).run()
        (out.run_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "spec": str(cfg.spec),
                    "docs": str(cfg.docs),
                    "profile": profile,
                    "strategy": strategy,
                    "target_config": str(target_config),
                    "local_target": local_target,
                    "local_reference": local_reference,
                    "base_url": None if local_target else base_url,
                    "reference_url": None if local_reference else reference_url,
                    "max_tokens": max_tokens,
                },
                indent=1,
            )
        )
        return out


@app.command()
def run(
    spec: Annotated[
        str | None, typer.Option(help="OpenAPI file or URL (default: from target config)")
    ] = None,
    docs: Annotated[str | None, typer.Option(help="Requirement docs directory or file")] = None,
    profile: Annotated[
        str, typer.Option(help="free | mixed | premium | simulated")
    ] = os.environ.get("AGENTQA_PROFILE", "simulated"),
    strategy: Annotated[str, typer.Option(help="S0 | S1 | S2 | S3 | S3-no-<mechanism>")] = "S3",
    target_config: Annotated[
        Path, typer.Option(help="Target auth/sandbox config")
    ] = DEFAULT_TARGET,
    base_url: Annotated[str | None, typer.Option(help="Target base URL")] = None,
    reference_url: Annotated[
        str | None, typer.Option(help="Known-good build used to verify generated tests")
    ] = None,
    local_target: Annotated[
        str | None, typer.Option(help="Start the bundled target locally: clean | all | B01,B04")
    ] = None,
    local_reference: Annotated[
        bool, typer.Option(help="Start a clean local build as the reference")
    ] = False,
    cache_mode: Annotated[str | None, typer.Option(help="off | read_write | replay_only")] = None,
    run_id: Annotated[str | None, typer.Option()] = None,
    max_tokens: Annotated[int | None, typer.Option(help="Token budget override")] = None,
) -> None:
    """Plan, generate, execute and triage tests for an API; write the report."""
    out = execute_run(
        spec=spec,
        docs=docs,
        profile=profile,
        strategy=strategy,
        target_config=target_config,
        base_url=base_url,
        reference_url=reference_url,
        local_target=local_target,
        local_reference=local_reference,
        cache_mode=cache_mode,
        run_id=run_id,
        max_tokens=max_tokens,
    )
    r = out.report
    typer.echo(
        f"run {out.run_id}: {r.outcome_counts()} | findings {len(r.findings)} "
        f"({len(r.product_bugs())} product bugs, {len(r.blocking())} blocking)"
    )
    typer.echo(
        f"tokens {r.cost.get('total_tokens')} | list-equivalent ${r.cost.get('cost_usd_list_equivalent', 0):.4f} "
        f"| trace {r.trace_id}"
    )
    typer.echo(f"report: {out.paths['report_html']}\nsummary: {out.paths['summary_html']}")
    if r.aborted:
        typer.echo(f"ABORTED: {r.aborted}")


@app.command()
def report(
    run_id: str,
    fmt: Annotated[str, typer.Option("--format", help="md | html | summary | json")] = "md",
) -> None:
    """Print (or locate) a run's report."""
    run_dir = agentqa_home() / "runs" / run_id
    name = {
        "md": "report.md",
        "html": "report.html",
        "summary": "executive_summary.md",
        "json": "report.json",
    }[fmt]
    path = run_dir / name
    if not path.exists():
        raise typer.BadParameter(f"no {name} for run {run_id}")
    typer.echo(path.read_text() if fmt in ("md", "summary") else str(path))


@app.command()
def replay(run_id: str) -> None:
    """Re-run a previous run with the LLM cache in replay_only mode (no API calls, deterministic)."""
    cfg = json.loads((agentqa_home() / "runs" / run_id / "run_config.json").read_text())
    out = execute_run(
        spec=cfg["spec"],
        docs=cfg["docs"],
        profile=cfg["profile"],
        strategy=cfg["strategy"],
        target_config=Path(cfg["target_config"]),
        base_url=cfg["base_url"],
        reference_url=cfg["reference_url"],
        local_target=cfg["local_target"],
        local_reference=cfg["local_reference"],
        cache_mode="replay_only",
        run_id=f"{run_id}-replay",
        max_tokens=cfg["max_tokens"],
    )
    typer.echo(
        f"replayed as {out.run_id}: {out.report.outcome_counts()} findings {len(out.report.findings)}"
    )


@app.command()
def runs(limit: int = 20) -> None:
    """List recent runs."""
    from agentqa.store import Store

    for r in Store().runs(limit):
        typer.echo(
            f"{r['run_id']}  {r['status']:<8} {r['profile']:<9} {r['strategy']:<6} {r['summary'] or ''}"
        )


@app.command()
def killswitch(
    release: Annotated[bool, typer.Option(help="Remove the kill switch")] = False,
    reason: str = "manual stop",
) -> None:
    """Stop every running pipeline at its next check (or release the switch)."""
    from agentqa.guards import killswitch as ks

    if release:
        ks.release()
        typer.echo("kill switch released")
    else:
        typer.echo(f"kill switch engaged: {ks.engage(reason)}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8088) -> None:
    """Web UI: run history, run detail, eval scoreboard."""
    import uvicorn

    uvicorn.run("agentqa.api.app:app", host=host, port=port, log_level="warning")


@app.command("eval")
def eval_cmd(
    matrix: Annotated[bool, typer.Option(help="Profile matrix (strategy S3)")] = False,
    profiles: Annotated[
        str, typer.Option(help="Comma-separated profiles for --matrix")
    ] = "simulated",
    strategies: Annotated[bool, typer.Option(help="S0-S3 strategy comparison")] = False,
    ablate: Annotated[bool, typer.Option(help="Also run S3 ablations")] = False,
    seeds: Annotated[int, typer.Option(help="Runs per configuration (>=3 for variance)")] = 3,
    memory: Annotated[
        bool, typer.Option(help="Run 1 vs run 2 (lessons) vs run 3 (incremental)")
    ] = False,
    extras: Annotated[bool, typer.Option(help="Injection set + judge calibration")] = False,
    perf: Annotated[bool, typer.Option(help="Performance A/B benchmark (P01-P06)")] = False,
    perf_iterations: int = 3,
    gate_check: Annotated[
        bool, typer.Option("--gate", help="CI regression gate (replay mode)")
    ] = False,
    baseline: Path = REPO_ROOT / "evals/baselines/main.json",
    tag: str = "",
) -> None:
    """Run eval suites and write results/<date>-<commit>.json and RESULTS.md."""
    from agentqa.evals import run as ev
    from agentqa.evals.report import render_results

    if gate_check:
        ok, detail = ev.gate(baseline)
        typer.echo(json.dumps(detail, indent=1, default=str))
        summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_file:
            with open(summary_file, "a", encoding="utf-8") as fh:
                fh.write("## AgentQA replay gate\n\n| check | ok |\n|---|---|\n")
                fh.writelines(f"| {k} | {v} |\n" for k, v in detail["checks"].items())
        raise typer.Exit(0 if ok else 1)
    payload: dict[str, Any] = {}
    if matrix:
        payload["profiles"] = profiles.split(",")
        payload["matrix"] = ev.profile_matrix(profiles.split(","), seeds, log=typer.echo)
    if strategies:
        payload["strategies"] = ev.strategy_comparison(seeds, ablate, log=typer.echo)
    if memory:
        payload["memory"] = ev.memory_effect(log=typer.echo)
    if extras:
        payload["injection"] = ev.injection_suite()
        payload["judge_calibration"] = ev.judge_calibration()
    func_path = ev.write_results(payload, tag or "functional") if payload else None
    perf_path = None
    if perf:
        perf_path = ev.write_results(
            ev.perf_benchmark(iterations=perf_iterations, log=typer.echo), "perf"
        )
    results = sorted((REPO_ROOT / "results").glob("*.json"))
    func_path = func_path or next((p for p in reversed(results) if "perf" not in p.name), None)
    perf_path = perf_path or next((p for p in reversed(results) if "perf" in p.name), None)
    typer.echo(f"wrote {render_results(func_path, perf_path)}")


perf_app = typer.Typer(help="Performance tests: workload model, load runs, A/B, diagnosis.")
app.add_typer(perf_app, name="perf")


@perf_app.command("run")
def perf_run(
    test_type: Annotated[
        str, typer.Option("--type", help="smoke | load | stress | spike | soak")
    ] = "smoke",
    perf_bugs: Annotated[str, typer.Option(help="Local build: clean or P01,P04")] = "clean",
    iterations: Annotated[int | None, typer.Option(help="Default from config (3)")] = None,
    seed_orders: Annotated[
        int | None, typer.Option(help="Override the workload's data volume")
    ] = None,
    profile: str = "simulated",
    target_config: Path = DEFAULT_TARGET,
) -> None:
    """Run one load test against a local build and check SLOs."""
    from agentqa.perf import analysis
    from agentqa.perf.charts import load_charts
    from agentqa.perf.pipeline import PerfSession
    from agentqa.target_config import load_target

    target = load_target(target_config)
    s = PerfSession(target, target.spec_path, target.docs_path, profile=profile)
    bugs = [] if perf_bugs == "clean" else [b.strip() for b in perf_bugs.split(",")]
    run = s.run_build(perf_bugs, bugs, test_type, iterations, seed_orders)
    users, rps, p95 = analysis.curve(analysis.summarize(run))
    typer.echo(
        f"{test_type} on {perf_bugs}: users {users} rps {rps} p95 {p95} knee {analysis.knee(users, rps, p95)}"
    )
    for v in analysis.slo_verdicts(run, s.workload()):
        typer.echo(
            f"  SLO {v['slo']} ({v['source']}): {v['actual']} vs {v['limit']} -> {'met' if v['met'] else 'VIOLATED'}"
        )
    typer.echo(
        f"chart: {load_charts({'candidate': run}, s.out_dir / f'chart-{perf_bugs}-{test_type}.png', test_type)}"
    )


@perf_app.command("ab")
def perf_ab(
    test_type: Annotated[str, typer.Option("--type")] = "smoke",
    candidate: Annotated[
        str, typer.Option(help="Perf defects in the candidate build, e.g. P01")
    ] = "P01",
    iterations: Annotated[int | None, typer.Option()] = None,
    seed_orders: Annotated[int | None, typer.Option()] = 20000,
    profile: str = "simulated",
    target_config: Path = DEFAULT_TARGET,
    fail_on_regression: Annotated[
        bool, typer.Option(help="Exit 1 if the candidate regresses (CI gate)")
    ] = True,
    baseline_root: Annotated[
        Path | None, typer.Option(help="Checkout of the baseline code (e.g. main) for PR A/B")
    ] = None,
) -> None:
    """Relative A/B: clean vs candidate back to back on this host, with a measured noise band."""
    from agentqa.perf import analysis
    from agentqa.perf.pipeline import PerfSession, perf_report
    from agentqa.target_config import load_target

    target = load_target(target_config)
    s = PerfSession(target, target.spec_path, target.docs_path, profile=profile)
    bugs = [] if candidate == "clean" else [b.strip() for b in candidate.split(",")]
    base = s.run_build("clean", [], test_type, iterations, seed_orders, code_root=baseline_root)
    base2 = s.run_build(
        "clean-repeat", [], test_type, iterations, seed_orders, code_root=baseline_root
    )
    band = analysis.noise_band(base, base2)
    cand = s.run_build(f"candidate-{candidate}", bugs, test_type, iterations, seed_orders)
    res = s.ab(base, cand, band, f"ab-{candidate}")
    report = perf_report(
        [res], s.workload(), s.workload_path, s.out_dir / f"perf_report-{candidate}.md"
    )
    summary = res.summary()
    typer.echo(
        json.dumps(
            {
                k: summary[k]
                for k in ("regressed", "server_signals", "diagnosis", "rule_diagnosis", "fix")
            },
            indent=1,
        )
    )
    typer.echo(f"report: {report}")
    if fail_on_regression and summary["regressed"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
