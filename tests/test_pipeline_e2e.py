"""End-to-end: simulated models, all-bugs target, clean reference build."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentqa.obs import tracing
from agentqa.orchestrator.pipeline import Pipeline, RunConfig
from agentqa.target_config import load_target
from target_api.launcher import TargetServer

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parent.parent
ALL = [f"B{i:02d}" for i in range(1, 13)]


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    with TargetServer(bugs=ALL) as tgt, TargetServer() as ref:
        out = Pipeline(
            RunConfig(
                spec=ROOT / "target_api/openapi.json",
                docs=ROOT / "target_api/docs",
                target=load_target(),
                base_url=tgt.base_url,
                reference_url=ref.base_url,
                server_log=tgt.log_lines,
                cache_mode="off",
                out_root=tmp_path,
            )
        ).run()
    r = out.report
    assert r.simulated and not r.aborted
    assert r.outcome_counts()["failed"] >= 8 and len(r.product_bugs()) >= 6
    assert any(vt == "T0" for vt in r.tiers.values()) and any(
        vt in ("T1", "T2") for vt in r.tiers.values()
    )
    assert r.quarantined_chunks and all("poisoned" in c for c in r.quarantined_chunks)
    for f in r.findings:
        assert f.evidence and f.repro_curl.startswith("curl")
    for name in (
        "report.md",
        "report.html",
        "executive_summary.md",
        "executive_summary.html",
        "spans.jsonl",
        "stats.json",
    ):
        assert (out.run_dir / name).exists(), name
    # trace hierarchy: run > stage/agent > llm_call, with cost attributes, all in one trace
    spans = [json.loads(line) for line in (out.run_dir / "spans.jsonl").read_text().splitlines()]
    kinds = {s["attributes"].get("agentqa.span_kind") for s in spans}
    assert {"run", "stage", "agent", "llm_call", "tool_call"} <= kinds
    assert len({s["trace_id"] for s in spans}) == 1
    llm = [s for s in spans if s["attributes"].get("agentqa.span_kind") == "llm_call"]
    assert all("agentqa.cost_usd_list_equivalent" in s["attributes"] for s in llm)
    by_id = {s["span_id"]: s for s in spans}
    # walk from a generator llm call up to the run span
    gen = next(s for s in llm if s["attributes"].get("agentqa.agent") == "generator")
    chain = []
    node = gen
    while node:
        chain.append(node["attributes"].get("agentqa.span_kind"))
        node = by_id.get(node["parent_id"])
    assert chain[0] == "llm_call" and chain[-1] == "run"
    assert out.stats["hallucination"]["checked"] > 0
    assert tracing.finished_spans(r.trace_id)
