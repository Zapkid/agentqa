"""Hand-written correctness checks, one per seeded defect.

Each check asserts the *correct* behaviour from the requirement docs, so it passes on the clean
build and fails when its defect is switched on. test_seeded_bugs.py proves both directions,
which verifies the benchmark itself.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable
from typing import Any

from target_api.app.seed import sid

ADMIN = {"Authorization": "Bearer tok-admin"}
STAFF = {"Authorization": "Bearer tok-staff"}
ALICE = {"Authorization": "Bearer tok-alice"}
BOB = {"Authorization": "Bearer tok-bob"}
SECRET = "change-me-local-only"

Client = Any  # fastapi TestClient or httpx.Client with base_url


def _order(c: Client, headers: dict[str, str], items: list[dict[str, Any]], **extra: Any) -> Any:
    return c.post("/orders", json={"items": items}, headers={**headers, **extra})


def check_b01(c: Client) -> None:
    r = _order(c, ALICE, [{"product_id": sid("product:WIDGET"), "quantity": -2}])
    assert r.status_code == 422, r.text


def check_b02(c: Client) -> None:
    all_rows = c.get("/orders", params={"page_size": 30}, headers=ADMIN).json()["items"]
    page2 = c.get("/orders", params={"page": 2, "page_size": 10}, headers=ADMIN).json()["items"]
    assert [o["id"] for o in page2] == [o["id"] for o in all_rows[10:20]]


def check_b03(c: Client) -> None:
    r = _order(
        c,
        ALICE,
        [
            {"product_id": sid("product:BOLT"), "quantity": 1},
            {"product_id": sid("product:CABLE"), "quantity": 1},
        ],
    )
    assert r.status_code == 201, r.text
    lines = {i["unit_price"]: i["line_total"] for i in r.json()["items"]}
    assert lines == {"0.125": "0.13", "1.005": "1.01"}
    assert r.json()["total"] == "1.14"


def check_b04(c: Client) -> None:
    r = c.get(f"/orders/{sid('order:bob-pending')}", headers=ALICE)
    assert r.status_code == 404


def check_b05(c: Client) -> None:
    r = c.patch(f"/products/{sid('product:WIDGET')}", json={"price": "0.01"}, headers=STAFF)
    assert r.status_code == 403


def check_b06(c: Client) -> None:
    items = [{"product_id": sid("product:GADGET"), "quantity": 1}]
    first = _order(c, ALICE, items, **{"Idempotency-Key": "idem-proof-1"})
    second = _order(c, ALICE, items, **{"Idempotency-Key": "idem-proof-1"})
    assert first.status_code == 201
    assert second.json()["id"] == first.json()["id"]


def check_b07(c: Client) -> None:
    r = c.post(f"/orders/{sid('order:bob-cancelled')}/ship", headers=STAFF)
    assert r.status_code == 409


def check_b08(c: Client) -> None:
    r = c.get("/orders/not-a-uuid", headers=ADMIN)
    assert r.status_code in (404, 422)


def check_b09(c: Client) -> None:
    r = c.get("/orders", params={"status": "teleported"}, headers=ADMIN)
    assert r.status_code == 422


def check_b10(c: Client) -> None:
    r = c.get(f"/customers/{sid('customer:alice')}", headers=ADMIN)
    assert set(r.json()) == {"id", "name", "email", "created_at"}


def check_b11(c: Client) -> None:
    body = json.dumps(
        {
            "event_id": "evt-proof",
            "order_id": sid("order:alice-pending"),
            "amount": "19.99",
            "status": "succeeded",
        }
    ).encode()
    bad = hmac.new(b"wrong-secret", body, hashlib.sha256).hexdigest()
    r = c.post(
        "/webhooks/payment",
        content=body,
        headers={"X-Signature": bad, "Content-Type": "application/json"},
    )
    assert r.status_code == 401


def check_b12(c: Client) -> None:
    # alice-pending was created at 2026-03-01T09:00:00Z; 10:30+02:00 is 08:30Z, so it must be included.
    r = c.get(
        "/orders",
        params={"created_from": "2026-03-01T10:30:00+02:00", "page_size": 100},
        headers=ADMIN,
    )
    assert sid("order:alice-pending") in [o["id"] for o in r.json()["items"]]


CHECKS: dict[str, Callable[[Client], None]] = {
    f"B{i:02d}": fn
    for i, fn in enumerate(
        [
            check_b01,
            check_b02,
            check_b03,
            check_b04,
            check_b05,
            check_b06,
            check_b07,
            check_b08,
            check_b09,
            check_b10,
            check_b11,
            check_b12,
        ],
        start=1,
    )
}


def valid_signature(body: bytes, secret: str = SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
