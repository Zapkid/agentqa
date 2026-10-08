"""The Tasks API example: a Swagger 2.0 contract with X-API-Key auth, and AgentQA run against it."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agentqa.ingest.openapi import HTTP_METHODS, load_spec, parse_endpoints
from agentqa.orchestrator.pipeline import Pipeline, RunConfig
from agentqa.target_config import TargetConfig, load_target
from examples.tasks_api import app as tasks
from target_api.launcher import TargetServer

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples/tasks_api"
APP = "examples.tasks_api.app:app"
ALICE, BOB, ADMIN = ({"X-API-Key": k} for k in ("key-alice", "key-bob", "key-admin"))


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(tasks, "BUGS", set())
    tasks._seed()
    return TestClient(tasks.app)


def _bugs(monkeypatch: pytest.MonkeyPatch, *ids: str) -> None:
    monkeypatch.setattr(tasks, "BUGS", set(ids))


def test_swagger_matches_the_routes() -> None:
    documented = {e.id for e in parse_endpoints(load_spec(EXAMPLE / "swagger.json"))}
    implemented = {
        f"{m.upper()} {path}"
        for path, ops in tasks.app.openapi()["paths"].items()
        for m in ops
        if m in HTTP_METHODS
    }
    assert documented == implemented


def test_target_config_uses_api_key_header() -> None:
    target = load_target(EXAMPLE / "agentqa_target.yaml")
    assert target.headers("admin") == {"X-API-Key": "key-admin"}
    assert target.auth.roles["customer"].customer_id == tasks.USERS["key-alice"]["id"]
    bearer = TargetConfig(name="x", spec="s", auth={"roles": {"a": {"token": "t"}}})  # type: ignore[arg-type]
    assert bearer.headers("a") == {"Authorization": "Bearer t"}


def test_authentication(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/tasks").status_code == 401
    assert client.get("/tasks", headers={"X-API-Key": "nope"}).status_code == 401
    assert client.get("/tasks", headers=ALICE).status_code == 200


def test_ownership_is_enforced(client: TestClient) -> None:
    created = client.post("/tasks", json={"title": "mine"}, headers=ALICE)
    assert created.status_code == 201
    tid = created.json()["id"]
    assert client.get(f"/tasks/{tid}", headers=BOB).status_code == 404
    assert client.get(f"/tasks/{tid}", headers=ADMIN).status_code == 200
    assert tid not in {
        t["id"] for t in client.get("/tasks?page_size=100", headers=BOB).json()["items"]
    }


def test_validation(client: TestClient) -> None:
    assert client.post("/tasks", json={"title": ""}, headers=ALICE).status_code == 422
    assert (
        client.post("/tasks", json={"title": "x", "priority": "urgent"}, headers=ALICE).status_code
        == 422
    )
    assert (
        client.post(
            "/tasks", json={"title": "x", "due_date": "2001-01-01"}, headers=ALICE
        ).status_code
        == 422
    )
    assert client.get("/tasks?status=bogus", headers=ALICE).status_code == 422
    assert client.get("/tasks/not-a-uuid", headers=ALICE).status_code == 422


def test_completed_tasks_are_final(client: TestClient) -> None:
    tid = client.post("/tasks", json={"title": "t"}, headers=ALICE).json()["id"]
    assert client.post(f"/tasks/{tid}/complete", headers=ALICE).json()["status"] == "done"
    assert client.post(f"/tasks/{tid}/complete", headers=ALICE).status_code == 409
    assert client.patch(f"/tasks/{tid}", json={"title": "new"}, headers=ALICE).status_code == 409


def test_pages_are_contiguous(client: TestClient) -> None:
    full = client.get("/tasks?page=1&page_size=30", headers=ADMIN).json()["items"]
    page2 = client.get("/tasks?page=2&page_size=10", headers=ADMIN).json()["items"]
    assert [t["id"] for t in page2] == [t["id"] for t in full[10:20]]


@pytest.mark.parametrize("bug", ["T01", "T02", "T03", "T04"])
def test_each_seeded_bug_changes_behaviour(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, bug: str
) -> None:
    _bugs(monkeypatch, bug)
    tid = client.post("/tasks", json={"title": "t"}, headers=ALICE).json()["id"]
    if bug == "T01":
        assert client.get(f"/tasks/{tid}", headers=BOB).status_code == 200
    elif bug == "T02":
        assert client.get("/tasks?status=bogus", headers=ALICE).status_code == 200
    elif bug == "T03":
        client.post(f"/tasks/{tid}/complete", headers=ALICE)
        assert client.post(f"/tasks/{tid}/complete", headers=ALICE).status_code == 200
    else:
        full = client.get("/tasks?page_size=30", headers=ADMIN).json()["items"]
        page2 = client.get("/tasks?page=2&page_size=10", headers=ADMIN).json()["items"]
        assert [t["id"] for t in page2] != [t["id"] for t in full[10:20]]


def _run(bugs: list[str], out: Path) -> list[str]:
    with TargetServer(bugs=bugs, app=APP) as srv:
        result = Pipeline(
            RunConfig(
                spec=EXAMPLE / "swagger.json",
                docs=EXAMPLE / "docs",
                target=load_target(EXAMPLE / "agentqa_target.yaml"),
                base_url=srv.base_url,
                cache_mode="off",
                out_root=out,
                phoenix_base=None,
            )
        ).run()
    assert result.report.simulated and not result.report.aborted
    assert result.report.outcome_counts()["error"] == 0
    return sorted(f.title for f in result.report.findings)


@pytest.mark.slow
def test_agentqa_finds_the_spec_visible_bugs(tmp_path: Path) -> None:
    titles = _run(["T02", "T04"], tmp_path)
    assert any("enum" in t for t in titles) and any("contiguous" in t for t in titles)
    assert len(titles) == 2


@pytest.mark.slow
def test_clean_build_has_no_findings(tmp_path: Path) -> None:
    assert _run([], tmp_path) == []
