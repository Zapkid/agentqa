from __future__ import annotations

from pathlib import Path

import pytest

from agentqa import config
from agentqa.dispatch import savings, strategies
from agentqa.dispatch.cascade import DelegationLedger, Verifier, run_cascade
from agentqa.dispatch.features import difficulty, features
from agentqa.dispatch.learned import RoutingStats
from agentqa.dispatch.policy import choose_tier
from agentqa.ingest.openapi import load_spec, parse_endpoints
from agentqa.llm.types import BudgetExceeded, ProviderUnavailable
from agentqa.models import GeneratedTest, RequestDecl, TestIntent
from agentqa.orchestrator.graph import Supervisor, Task, TaskGraph
from agentqa.store import Store

ROOT = Path(__file__).resolve().parent.parent
EPS = parse_endpoints(load_spec(ROOT / "target_api" / "openapi.json"))


def intent(
    iid: str = "i-1", category: str = "validation", risk: int = 3, endpoint: str = "GET /orders"
) -> TestIntent:
    return TestIntent(
        id=iid,
        endpoint=endpoint,
        category=category,
        title=f"title {iid}",
        risk=risk,
        rationale="r",
        expected_behavior="e",
        source_refs=["spec:GET /orders"],
    )


GOOD = GeneratedTest(
    intent_id="i-1",
    test_name="test_good_case",
    code="def test_good_case(client, auth):\n    assert client.get('/orders', headers=auth['admin']).status_code == 200\n",
    requests_made=[RequestDecl(method="GET", path="/orders", expected_status=[200])],
    confidence=0.9,
)
BAD = GOOD.model_copy(
    update={
        "code": GOOD.code.replace("'/orders'", "'/v1/orders'"),
        "requests_made": [RequestDecl(method="GET", path="/v1/orders")],
    }
)


# ------------------------------------------------------------------ supervisor


def test_graph_priorities_failure_isolation_and_spawn() -> None:
    order: list[str] = []
    g = TaskGraph()

    def work(name: str, fail: bool = False) -> str:
        order.append(name)
        if fail:
            raise ValueError("boom")
        return name

    def spawn(_: str) -> list[Task]:
        return [
            Task("low", "x", lambda: work("low"), deps=["root"], priority=1),
            Task("high", "x", lambda: work("high"), deps=["root"], priority=9),
            Task("bad", "x", lambda: work("bad", True), deps=["root"], priority=5),
            Task("after_bad", "x", lambda: work("after_bad"), deps=["bad"]),
            Task(
                "fanin",
                "x",
                lambda: work("fanin"),
                deps=["low", "high", "bad"],
                tolerate_failures=True,
            ),
        ]

    g.add(Task("root", "x", lambda: work("root"), spawn=spawn))
    Supervisor(g, concurrency=1).run()
    assert order[:4] == ["root", "high", "bad", "low"]
    assert g.tasks["bad"].status == "failed" and "boom" in (g.tasks["bad"].error or "")
    assert g.tasks["after_bad"].status == "skipped"
    assert g.tasks["fanin"].status == "done"


def test_graph_checkpoint_resume(tmp_path: Path) -> None:
    store = Store(tmp_path / "db.sqlite")
    calls: list[str] = []
    g = TaskGraph()
    g.add(Task("a", "x", lambda: calls.append("a") or 41))
    g.add(Task("b", "x", lambda: (_ for _ in ()).throw(RuntimeError("crash")), deps=["a"]))
    Supervisor(g, store=store, run_id="r1").run()
    assert g.tasks["b"].status == "failed"
    g2 = TaskGraph()
    g2.add(Task("a", "x", lambda: calls.append("a2") or 0))
    g2.add(Task("b", "x", lambda: g2.result("a") + 1, deps=["a"]))
    Supervisor(g2, store=store, run_id="r1").run()
    assert calls == ["a"] and g2.result("b") == 42  # 'a' resumed from its checkpoint, not re-run


def test_graph_abort_runs_only_always_run() -> None:
    g = TaskGraph()
    ran: list[str] = []
    g.add(Task("a", "x", lambda: ran.append("a")))
    g.add(Task("b", "x", lambda: ran.append("b"), deps=["a"]))
    g.add(
        Task(
            "report",
            "x",
            lambda: ran.append("report"),
            deps=["b"],
            tolerate_failures=True,
            always_run=True,
        )
    )

    def before(t: Task) -> None:
        if t.id == "b":
            raise BudgetExceeded("tokens")

    sup = Supervisor(g, before_task=before)
    sup.run()
    assert ran == ["a", "report"] and sup.aborted and "tokens" in sup.aborted
    assert g.tasks["b"].status == "skipped"


# ------------------------------------------------------------------ policy and learning


def test_difficulty_is_transparent() -> None:
    ep = next(e for e in EPS if e.id == "POST /orders")
    easy = features(intent(category="validation", risk=2), ep)
    hard = features(intent(category="money/precision", risk=5), ep)
    assert difficulty(hard) > difficulty(easy)
    assert difficulty(
        features(intent(category="state-transition"), ep, prior_failures=3)
    ) > difficulty(features(intent(category="state-transition"), ep))


def test_policy_static_cascade_forced(tmp_path: Path) -> None:
    prof = config.profile("simulated")
    d = config.dispatch()
    assert choose_tier("t", "validation", 0.2, d, prof, None).tier == "T1"
    assert (
        choose_tier(
            "t", "money/precision", 0.9, d.with_overrides(learned_routing=False), prof, None
        ).tier
        == "T2"
    )
    static = choose_tier("t", "x", 0.9, d.with_overrides(cascade=False), prof, None)
    assert static.tier == "T1" and not static.can_escalate and "static" in static.reason
    assert choose_tier("t", "x", 0.1, d, prof, None, forced_tier="T2").tier == "T2"
    stats = RoutingStats(Store(tmp_path / "db.sqlite"))
    a = choose_tier("task-7", "auth", 0.3, d, prof, stats)
    assert a == choose_tier("task-7", "auth", 0.3, d, prof, stats)  # reproducible
    for _ in range(30):
        stats.update("auth", "T1", success=False)
    starts = [choose_tier(f"task-{n}", "auth", 0.3, d, prof, stats).tier for n in range(40)]
    assert starts.count("T2") > 30  # learned that T1 keeps failing on this task type


# ------------------------------------------------------------------ cascade


def _cascade(script: list[GeneratedTest | Exception], start: str = "T1", can_escalate: bool = True):  # type: ignore[no-untyped-def]
    calls: list[tuple[str, str | None]] = []

    def generate(it: TestIntent, tier: str, feedback: str | None) -> GeneratedTest:
        calls.append((tier, feedback))
        item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    out = run_cascade(intent(), start, can_escalate, generate, Verifier(EPS), lambda: 0)
    return out, calls


def test_cascade_first_try_ok() -> None:
    out, calls = _cascade([GOOD])
    assert (
        out.validated and out.path == ["T1:first:ok"] and not out.hallucinated and len(calls) == 1
    )


def test_cascade_repair_then_ok_records_pre_repair() -> None:
    out, calls = _cascade([BAD, GOOD])
    assert out.validated and out.validated.repaired and out.hallucinated
    assert calls[1][1] and "unknown_path" in calls[1][1]  # feedback carries the violation


def test_cascade_escalates_with_history() -> None:
    out, calls = _cascade([BAD, BAD, GOOD])
    assert out.validated and out.validated.tier == "T2"
    assert [c[0] for c in calls] == ["T1", "T1", "T2"]
    assert calls[2][1] and calls[2][1].count("rejected") == 2


def test_cascade_quarantines_and_handles_generation_errors() -> None:
    out, _ = _cascade([ProviderUnavailable("down"), BAD], can_escalate=False)
    assert out.validated is None and out.uncovered_reason and "quarantined" in out.uncovered_reason


def test_delegation_ledger(tmp_path: Path) -> None:
    led = DelegationLedger(Store(tmp_path / "db.sqlite"), "run-x")
    led.record(
        task_id="a",
        task_type="validation",
        features={},
        start_tier="T1",
        reason="r",
        est_tokens=10,
        actual_tokens=12,
        cost_usd=0.0,
        verifier_outcome="ok",
        escalation_path=["T1:first:unknown_path", "T1:repair:x", "T2:first:ok"],
        final_result="validated",
    )
    led.record(
        task_id="b",
        task_type="validation",
        features={},
        start_tier="T0",
        reason="r",
        est_tokens=0,
        actual_tokens=0,
        cost_usd=0.0,
        verifier_outcome="ok",
        escalation_path=["T0:ok"],
        final_result="validated",
    )
    s = led.summary()
    assert s["escalations"] == 1 and s["tasks"] == 2
    assert len(led.store.query("SELECT * FROM delegations WHERE run_id='run-x'")) == 2


# ------------------------------------------------------------------ savings


def test_dedupe_allocate_batches() -> None:
    a = intent("a", risk=4)
    dup = a.model_copy(update={"id": "a2", "title": "Check that title a", "risk": 3})
    b = intent("b", category="money/precision", risk=5, endpoint="POST /orders")
    kept, removed = savings.dedupe([a, dup, b], 0.6)
    assert [i.id for i in kept] == ["a", "b"] and [i.id for i in removed] == ["a2"]
    alloc = savings.allocate(
        [a, b, intent("c", risk=1)], tokens_available=9000, est={"T1": 3000, "T2": 3000}
    )
    assert alloc.tier_cap["b"] == "T2" and alloc.tier_cap["a"] == "T2"
    assert "c" in alloc.skipped and "budget" in alloc.skipped["c"]
    groups, singles = savings.batches([a, b, intent("c")], {"a": 0.2, "b": 0.9, "c": 0.3}, size=4)
    assert [len(g) for g in groups] == [2] and [s.id for s in singles] == ["b"]


def test_strategies() -> None:
    s0, s3 = strategies.get("S0"), strategies.get("S3")
    assert s0.forced_tier == "T2" and not any(s0.mechanisms.values())
    assert all(s3.mechanisms.values())
    ab = strategies.get("S3-no-batching")
    assert ab.mechanisms["batching"] is False and ab.mechanisms["cascade"] is True
    with pytest.raises(KeyError):
        strategies.get("S9")
