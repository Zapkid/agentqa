"""Drive the MCP server over stdio with the official MCP client (slow: runs a real pipeline)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parent.parent


def _text(result: object) -> str:
    content = getattr(result, "content", [])
    return "".join(getattr(c, "text", "") for c in content)


def _items(result: object) -> list[object]:
    """A list return value arrives as one text content item per element."""
    return [json.loads(getattr(c, "text", "null")) for c in getattr(result, "content", [])]


async def _session_flow(home: Path) -> dict[str, object]:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentqa.mcp.server"],
        cwd=str(ROOT),
        env={**os.environ, "AGENTQA_HOME": str(home), "AGENTQA_PROFILE": "simulated"},
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = {t.name for t in (await session.list_tools()).tools}
        ingest = json.loads(_text(await session.call_tool("ingest_spec", {})))
        run = json.loads(_text(await session.call_tool("run_suite", {"local_target": "B04,B11"})))
        findings = _items(await session.call_tool("triage_run", {"run_id": run["run_id"]}))
        res = await session.read_resource(f"agentqa://runs/{run['run_id']}/summary")
        summary = "".join(getattr(c, "text", "") for c in res.contents)
        return {
            "tools": tools,
            "ingest": ingest,
            "run": run,
            "findings": findings,
            "summary": summary,
        }


def test_mcp_stdio_end_to_end(tmp_path: Path) -> None:
    out = anyio.run(_session_flow, tmp_path)
    assert {
        "ingest_spec",
        "plan_tests",
        "run_suite",
        "triage_run",
        "get_report",
        "list_runs",
        "perf_ab",
    } <= out["tools"]  # type: ignore[operator]
    assert any("poisoned" in c for c in out["ingest"]["quarantined_chunks"])  # type: ignore[index]
    assert out["run"]["product_bugs"] >= 1 and out["run"]["simulated"]  # type: ignore[index]
    assert out["findings"] and "repro_curl" in out["findings"][0]  # type: ignore[index]
    assert "Executive summary" in str(out["summary"])
