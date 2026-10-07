from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentqa.agents.reporter import RunReport, Uncovered, render
from agentqa.agents.triage import Triage, cluster_failures, curl_for
from agentqa.config import ModelRef
from agentqa.ingest.ingestor import ingest
from agentqa.ingest.vectorstore import VectorStore
from agentqa.llm.adapters.fake import FakeAdapter
from agentqa.llm.cache import DiskCache
from agentqa.llm.router import LLMClient, ProviderRegistry
from agentqa.llm.types import RawResponse
from agentqa.models import HttpExchange, RunResult, TestIntent, TestResult

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def bundle():  # type: ignore[no-untyped-def]
    b, _ = ingest(ROOT / "target_api" / "openapi.json", ROOT / "target_api" / "docs", VectorStore())
    return b


def intent(iid: str, endpoint: str = "GET /orders/{order_id}", origin: str = "llm") -> TestIntent:
    return TestIntent(
        id=iid,
        endpoint=endpoint,
        category="auth/permission",
        title=f"t {iid}",
        risk=5,
        rationale="r",
        expected_behavior="404 for another customer's order",
        source_refs=["spec:" + endpoint, "doc:02-orders#reading-an-order"],
        origin=origin,
    )  # type: ignore[arg-type]


def failed(
    name: str, iid: str, status: int = 200, msg: str = "AssertionError: assert 200 == 404"
) -> TestResult:
    ex = HttpExchange(
        method="GET",
        url="http://t/orders/x",
        path="/orders/x",
        status=status,
        request_headers={"authorization": "[REDACTED]"},
        response_body='{"id": "x"}',
    )
    return TestResult(
        test_name=name,
        intent_id=iid,
        outcome="failed",
        message=msg,
        exchanges=[ex],
        reruns=["failed", "failed", "failed"],
    )


def test_clustering_merges_same_symptom() -> None:
    pairs = [
        (failed("a", "i1"), intent("i1")),
        (failed("b", "i2"), intent("i2")),
        (failed("c", "i3", 500, "AssertionError: assert 500 < 500"), intent("i3")),
    ]
    assert len(cluster_failures(pairs)) == 2
    assert len(cluster_failures(pairs, enabled=False)) == 3
    t0 = [
        (
            failed("x", "t0-baduuid-a", 500, "assert 500 < 500"),
            intent("t0-baduuid-a", "GET /orders/{order_id}", "t0"),
        ),
        (
            failed("y", "t0-baduuid-b", 500, "assert 500 < 500"),
            intent("t0-baduuid-b", "GET /products/{product_id}", "t0"),
        ),
    ]
    assert len(cluster_failures(t0)) == 1  # same root cause across endpoints


def test_curl_repro() -> None:
    ex = HttpExchange(
        method="POST",
        url="http://h/orders",
        path="/orders",
        status=201,
        request_headers={"authorization": "[REDACTED]", "content-type": "application/json"},
        request_body='{"a": 1}',
    )
    c = curl_for(ex)
    assert c.startswith("curl -sS -X POST") and "Bearer $TOKEN" in c and "--data" in c


def _client(responses: list[str]) -> LLMClient:
    fake = FakeAdapter(script=[RawResponse(text=t) for t in responses])
    return LLMClient(
        [ModelRef(provider="simulated", model="sim-cheap")],
        registry=ProviderRegistry({"simulated": fake}),
        cache=DiskCache(mode="off"),
    )


def test_triage_cites_evidence_or_downgrades(bundle) -> None:  # type: ignore[no-untyped-def]
    run = RunResult(
        target="t",
        base_url="http://t",
        started_ts=0,
        ended_ts=1,
        results=[
            failed("a", "i1"),
            failed("b", "i2", 500, "AssertionError: other"),
            TestResult(
                test_name="flip",
                intent_id="i3",
                outcome="failed",
                message="x",
                reruns=["passed", "failed"],
            ),
            TestResult(
                test_name="blk", intent_id="i4", outcome="blocked", message="SandboxViolation"
            ),
        ],
        server_log=['{"path": "/orders/x", "status": 200}'],
    )
    intents = {i: intent(i) for i in ("i1", "i2", "i3", "i4")}
    good = {
        "classification": "product_bug",
        "severity": "high",
        "root_cause_hypothesis": "IDOR",
        "evidence_refs": ["exchange:a:0", "spec:GET /orders/{order_id}"],
        "confidence": 0.9,
        "title": "IDOR",
    }
    bogus = {**good, "evidence_refs": ["exchange:nope:0"], "title": "unsupported"}
    t = Triage(bundle)
    shared = _client([json.dumps(good), json.dumps(bogus)])
    findings = t.triage(run, intents, lambda tier: shared)
    by = {f.title: f for f in findings}
    assert by["IDOR"].classification == "product_bug" and {
        e.ref for e in by["IDOR"].evidence
    } == set(good["evidence_refs"])
    assert by["unsupported"].classification == "needs_review"
    assert {f.classification for f in findings if f.triaged_by == "rule"} == {"flaky", "test_bug"}
    assert t.stats["llm_calls"] == 2 and t.stats["downgraded"] == 1


def test_low_confidence_escalates(bundle) -> None:  # type: ignore[no-untyped-def]
    run = RunResult(
        target="t", base_url="http://t", started_ts=0, ended_ts=1, results=[failed("a", "i1")]
    )
    low = {
        "classification": "test_bug",
        "severity": "low",
        "root_cause_hypothesis": "?",
        "evidence_refs": ["exchange:a:0"],
        "confidence": 0.3,
        "title": "unsure",
    }
    high = {
        **low,
        "classification": "product_bug",
        "confidence": 0.9,
        "title": "sure",
        "severity": "high",
    }
    tiers: list[str] = []

    def client_for(tier: str) -> LLMClient:
        tiers.append(tier)
        return _client([json.dumps(low if tier == "T1" else high)])

    t = Triage(bundle)
    f = t.triage(run, {"i1": intent("i1")}, client_for)
    assert tiers == ["T1", "T2"] and f[0].title == "sure" and f[0].triaged_by == "T2"


def test_render_report(tmp_path: Path) -> None:
    i = intent("i1")
    r = RunReport(
        run_id="r",
        created_at="now",
        profile="simulated",
        strategy="S3",
        simulated=True,
        api_title="API",
        api_version="1",
        target="http://t",
        trace_id="abc",
        phoenix_url=None,
        endpoints=1,
        intents=[i],
        results=[TestResult(test_name="a", intent_id="i1", outcome="passed")],
        tiers={"i1": "T1"},
        findings=[],
        uncovered=[
            Uncovered(
                intent_id="i9",
                endpoint="GET /x",
                category="validation",
                risk=1,
                reason="budget: skipped",
            )
        ],
        guardrail_events=[],
        quarantined_chunks=["doc:poisoned#x"],
        hallucination={"checked": 1, "with_violations": 0, "quarantined": 0, "rate": 0.0},
        cost={"total_tokens": 10},
        delegation={},
    )
    paths = render(r, tmp_path)
    md = paths["report_md"].read_text()
    assert "SIMULATED" in md and "budget: skipped" in md and "doc:poisoned#x" in md
    html = paths["report_html"].read_text()
    assert "<table>" in html and "<script" not in html
    summary = paths["summary_md"].read_text()
    assert "estimate, not a measurement" in summary and "Formula" in summary
    assert r.effort_estimate()["hours"] == round(20 / 60, 1)
