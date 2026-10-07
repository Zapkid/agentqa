"""The benchmark verifies itself: each check passes clean and fails with its bug on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from target_api.app.seed import sid
from target_api.tests.checks import ADMIN, ALICE, CHECKS, STAFF, valid_signature

BUGS = yaml.safe_load((Path(__file__).parent.parent / "bugs.yaml").read_text())


def test_bugs_yaml_matches_checks() -> None:
    assert [b["id"] for b in BUGS] == list(CHECKS)
    for b in BUGS:
        assert {"id", "category", "endpoint", "description", "expected", "severity"} <= set(b)


@pytest.mark.parametrize("bug_id", list(CHECKS))
def test_check_passes_on_clean_build(client: TestClient, bug_id: str) -> None:
    CHECKS[bug_id](client)


@pytest.mark.parametrize("bug_id", list(CHECKS))
def test_check_fails_with_bug_on(
    client: TestClient, bug_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BUGS", bug_id)
    client.post("/__reset")
    with pytest.raises(AssertionError):
        CHECKS[bug_id](client)


@pytest.mark.parametrize("bug_id", list(CHECKS))
def test_other_bugs_do_not_trip_check(
    client: TestClient, bug_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single-bug isolation: every *other* bug on at once leaves this check passing."""
    monkeypatch.setenv("BUGS", ",".join(b for b in CHECKS if b != bug_id))
    client.post("/__reset")
    CHECKS[bug_id](client)


def test_happy_path_order_payment_invoice(client: TestClient) -> None:
    r = client.post(
        "/orders",
        json={"items": [{"product_id": sid("product:WIDGET"), "quantity": 2}]},
        headers=ALICE,
    )
    assert r.status_code == 201
    order = r.json()
    assert order["total"] == "39.98" and order["status"] == "pending"
    body = json.dumps(
        {"event_id": "evt-1", "order_id": order["id"], "amount": "39.98", "status": "succeeded"}
    ).encode()
    w = client.post(
        "/webhooks/payment",
        content=body,
        headers={"X-Signature": valid_signature(body), "Content-Type": "application/json"},
    )
    assert w.status_code == 200 and w.json() == {"result": "paid"}
    dup = client.post(
        "/webhooks/payment",
        content=body,
        headers={"X-Signature": valid_signature(body), "Content-Type": "application/json"},
    )
    assert dup.json() == {"result": "duplicate"}
    inv = client.get(f"/orders/{order['id']}/invoice", headers=ALICE)
    assert inv.status_code == 200 and inv.json()["amount"] == "39.98"
    shipped = client.post(f"/orders/{order['id']}/ship", headers=STAFF)
    assert shipped.json()["status"] == "shipped"


def test_auth_required(client: TestClient) -> None:
    assert client.get("/orders").status_code == 401
    assert client.get("/orders", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/customers", headers=ALICE).status_code == 403
    assert client.get("/customers", headers=ADMIN).status_code == 200


def test_customer_sees_only_own_orders(client: TestClient) -> None:
    items = client.get("/orders", params={"page_size": 100}, headers=ALICE).json()["items"]
    assert items and {o["customer_id"] for o in items} == {sid("customer:alice")}


def test_spec_is_identical_across_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    from target_api.app.main import app

    clean = json.dumps(app.openapi(), sort_keys=True)
    monkeypatch.setenv("BUGS", ",".join(CHECKS))
    app.openapi_schema = None
    assert json.dumps(app.openapi(), sort_keys=True) == clean
    committed = json.loads((Path(__file__).parent.parent / "openapi.json").read_text())
    assert json.dumps(committed, sort_keys=True) == clean
