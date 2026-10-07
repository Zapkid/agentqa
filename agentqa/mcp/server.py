"""F9 MCP server: AgentQA as tools for any MCP client (Claude Code, Cursor, ...).

Transports: stdio (default) and streamable HTTP.
    agentqa-mcp                       # stdio
    agentqa-mcp --http --port 8765    # streamable HTTP at http://127.0.0.1:8765/mcp

Tools: ingest_spec, plan_tests, run_suite, triage_run, get_report, list_runs, perf_ab.
Resources: agentqa://runs/{run_id}/report and agentqa://runs/{run_id}/summary.

Long-running tools run in a worker thread so the server stays responsive. Every guard applies
exactly as on the CLI: the sandbox, the budget, the kill switch and the load guard do not know
or care that the request came over MCP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anyio
from mcp.server.mcpserver import MCPServer

from agentqa.config import REPO_ROOT, agentqa_home

DEFAULT_TARGET = REPO_ROOT / "target_api" / "agentqa_target.yaml"

server = MCPServer(
    name="agentqa",
    instructions=(
        "AgentQA plans, generates, runs and triages API tests from an OpenAPI spec plus requirement docs. "
        "Start with ingest_spec, then run_suite (use local_target='all' to test the bundled demo API with every "
        "seeded bug). Read results with get_report or the agentqa://runs/{run_id}/report resource."
    ),
)


def _run_dir(run_id: str) -> Path:
    path = (agentqa_home() / "runs" / run_id).resolve()
    if not str(path).startswith(str((agentqa_home() / "runs").resolve())):
        raise ValueError("invalid run id")
    return path


@server.tool(
    description="Parse an OpenAPI spec and requirement docs; report endpoints and quarantined (prompt-injection) chunks."
)
async def ingest_spec(
    spec: str = str(REPO_ROOT / "target_api/openapi.json"),
    docs: str = str(REPO_ROOT / "target_api/docs"),
) -> dict[str, Any]:
    def work() -> dict[str, Any]:
        from agentqa.ingest.ingestor import ingest
        from agentqa.ingest.vectorstore import VectorStore

        bundle, counts = ingest(spec, docs, VectorStore())
        return {
            "api": f"{bundle.title} {bundle.version}",
            "spec_id": bundle.spec_id,
            "endpoints": [e.id for e in bundle.endpoints],
            "chunks": counts,
            "quarantined_chunks": [c.id for c in bundle.chunks if c.quarantined],
        }

    return await anyio.to_thread.run_sync(work)


@server.tool(
    description="Plan risk-ranked test intents for one endpoint (planner agent only; nothing is executed)."
)
async def plan_tests(
    endpoint: str = "POST /orders",
    profile: str = "simulated",
    spec: str = str(REPO_ROOT / "target_api/openapi.json"),
    docs: str = str(REPO_ROOT / "target_api/docs"),
) -> list[dict[str, Any]]:
    def work() -> list[dict[str, Any]]:
        from agentqa.agents.planner import Planner
        from agentqa.ingest.ingestor import ingest
        from agentqa.ingest.vectorstore import VectorStore
        from agentqa.llm.router import ModelRouter
        from agentqa.models import CATEGORIES

        store = VectorStore()
        bundle, _ = ingest(spec, docs, store)
        router = ModelRouter(profile)
        intents, _ = Planner(bundle, store).plan(
            bundle.endpoint(endpoint),
            router.for_role("planner"),
            list(CATEGORIES),
            spec_derived_covered=True,
        )
        return [
            i.model_dump(
                include={"id", "category", "title", "risk", "expected_behavior", "source_refs"}
            )
            for i in intents
        ]

    return await anyio.to_thread.run_sync(work)


@server.tool(
    description=(
        "Run the full pipeline (plan, generate, verify, execute in the sandbox, triage, report). Give base_url for a "
        "deployed sandbox target, or local_target='clean'|'all'|'B01,B04' to start the bundled demo API."
    )
)
async def run_suite(
    local_target: str | None = "all",
    base_url: str | None = None,
    profile: str = "simulated",
    strategy: str = "S3",
    local_reference: bool = True,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    def work() -> dict[str, Any]:
        from agentqa.cli import execute_run

        out = execute_run(
            spec=None,
            docs=None,
            profile=profile,
            strategy=strategy,
            target_config=DEFAULT_TARGET,
            base_url=base_url,
            reference_url=None,
            local_target=local_target,
            local_reference=local_reference,
            cache_mode=None,
            run_id=None,
            max_tokens=max_tokens,
        )
        r = out.report
        return {
            "run_id": out.run_id,
            "outcomes": r.outcome_counts(),
            "findings": len(r.findings),
            "product_bugs": len(r.product_bugs()),
            "blocking": [f.title for f in r.blocking()],
            "tokens": r.cost.get("total_tokens"),
            "cost_usd_list_equivalent": r.cost.get("cost_usd_list_equivalent"),
            "trace_id": r.trace_id,
            "simulated": r.simulated,
            "aborted": r.aborted,
            "report_resource": f"agentqa://runs/{out.run_id}/report",
        }

    return await anyio.to_thread.run_sync(work)


@server.tool(
    description="Findings of a run with classification, severity, evidence refs and a curl repro."
)
async def triage_run(run_id: str) -> list[dict[str, Any]]:
    data = json.loads((_run_dir(run_id) / "report.json").read_text())
    return [
        {
            k: f[k]
            for k in (
                "id",
                "title",
                "classification",
                "severity",
                "endpoint",
                "root_cause_hypothesis",
                "repro_curl",
                "confidence",
            )
        }
        | {"evidence": [e["ref"] for e in f["evidence"]]}
        for f in data["findings"]
    ]


@server.tool(
    description="A run's report: format 'summary' (executive summary) or 'technical' (Markdown)."
)
async def get_report(run_id: str, format: str = "summary") -> str:
    name = "executive_summary.md" if format == "summary" else "report.md"
    return (_run_dir(run_id) / name).read_text()


@server.tool(description="Recent runs with status and headline numbers.")
async def list_runs(limit: int = 10) -> list[dict[str, Any]]:
    from agentqa.store import Store

    return [
        {k: r[k] for k in ("run_id", "status", "profile", "strategy", "summary")}
        for r in Store().runs(limit)
    ]


@server.tool(
    description=(
        "Performance A/B: clean vs candidate build (e.g. 'P01') on the bundled demo API, with a measured noise band "
        "and a bottleneck diagnosis. Only runs against local sandbox builds (load guard)."
    )
)
async def perf_ab(
    candidate: str = "P01",
    test_type: str = "stress",
    iterations: int = 2,
    profile: str = "simulated",
) -> dict[str, Any]:
    def work() -> dict[str, Any]:
        from agentqa.perf import analysis
        from agentqa.perf.pipeline import PerfSession
        from agentqa.target_config import load_target

        target = load_target(DEFAULT_TARGET)
        s = PerfSession(target, target.spec_path, target.docs_path, profile=profile)
        base = s.run_build("clean", [], test_type, iterations, 20_000)
        repeat = s.run_build("clean-repeat", [], test_type, iterations, 20_000)
        band = analysis.noise_band(base, repeat)
        cand = s.run_build(candidate, [candidate], test_type, iterations, 20_000)
        return s.ab(base, cand, band, f"mcp-{candidate}").summary()

    return await anyio.to_thread.run_sync(work)


@server.resource(
    "agentqa://runs/{run_id}/report",
    mime_type="text/markdown",
    description="Technical report of a run",
)
def run_report(run_id: str) -> str:
    return (_run_dir(run_id) / "report.md").read_text()


@server.resource(
    "agentqa://runs/{run_id}/summary",
    mime_type="text/markdown",
    description="Executive summary of a run",
)
def run_summary(run_id: str) -> str:
    return (_run_dir(run_id) / "executive_summary.md").read_text()


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentQA MCP server")
    parser.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    from agentqa.obs.logging import configure_logging

    configure_logging("WARNING")
    if args.http:
        anyio.run(lambda: server.run_streamable_http_async(host=args.host, port=args.port))
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
