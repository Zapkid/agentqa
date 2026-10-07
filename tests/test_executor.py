from __future__ import annotations

from pathlib import Path

import pytest

from agentqa.executor.runner import Executor
from agentqa.models import GeneratedTest, RequestDecl, TestIntent, ValidatedTest
from agentqa.target_config import load_target
from target_api.launcher import TargetServer

pytestmark = pytest.mark.slow


def vt(name: str, code: str) -> ValidatedTest:
    intent = TestIntent(
        id=name.replace("_", "-"),
        endpoint="GET /orders",
        category="validation",
        title=name,
        risk=3,
        rationale="r",
        expected_behavior="e",
        source_refs=["spec:GET /orders"],
    )
    test = GeneratedTest(
        intent_id=intent.id,
        test_name=name,
        code=code,
        requests_made=[RequestDecl(method="GET", path="/orders")],
    )
    return ValidatedTest(intent=intent, test=test, tier="T1")


TESTS = [
    vt(
        "test_ok_case",
        "def test_ok_case(client, auth):\n    assert client.get('/orders', headers=auth['admin']).status_code == 200\n",
    ),
    vt(
        "test_fails_case",
        "def test_fails_case(client, auth):\n    assert client.get('/orders', headers=auth['admin']).status_code == 418\n",
    ),
    vt(
        "test_other_host_case",
        "def test_other_host_case(client):\n    client.get('http://example.com/')\n",
    ),
    vt(
        "test_raw_socket_case",
        "def test_raw_socket_case(client):\n    import socket\n    socket.create_connection(('1.1.1.1', 80), timeout=1)\n",
    ),
    vt(
        "test_delete_case",
        "def test_delete_case(client, auth):\n    client.delete('/orders', headers=auth['admin'])\n",
    ),
    vt("test_syntax_case", "def test_syntax_case(client):\n    return (:\n"),
]


def test_sandbox_blocks_and_logs(tmp_path: Path) -> None:
    target = load_target().model_copy(update={"sandbox": False})  # read-only: no mutations
    with TargetServer() as srv:
        run = Executor(target, reruns=3).run(
            TESTS, srv.base_url, tmp_path, server_log=srv.log_lines
        )
    by = {r.test_name: r for r in run.results}
    assert by["test_ok_case"].outcome == "passed"
    assert by["test_ok_case"].exchanges and by["test_ok_case"].exchanges[0].status == 200
    assert by["test_ok_case"].exchanges[0].request_headers.get("authorization") == "[REDACTED]"
    assert (
        by["test_fails_case"].outcome == "failed" and by["test_fails_case"].reruns == ["failed"] * 3
    )
    assert not by["test_fails_case"].flaky
    for name in ("test_other_host_case", "test_raw_socket_case", "test_delete_case"):
        assert by[name].outcome == "blocked", (name, by[name].message)
    assert by["test_syntax_case"].outcome == "error"
    kinds = sorted(e["kind"] for e in run.guardrail_events)
    assert kinds.count("host") >= 2 and "method" in kinds
    assert any('"route": "/orders"' in line for line in run.server_log)


def test_flaky_detection() -> None:
    from agentqa.models import TestResult

    r = TestResult(
        test_name="t", intent_id="i", outcome="failed", reruns=["passed", "failed", "passed"]
    )
    assert r.flaky
