"""Knowledge of the simulated models (ADR 0003).

The simulated planner "reads" the retrieved documents by keyword: an intent can only be planned
if the document chunk that states the rule was actually retrieved (so retrieval quality and the
injection quarantine matter). The simulated generator has a template per intent: what a
competent model would write after reading that rule. Mistakes are injected on top of this
with the declared error rates in ``agentqa/llm/simulated.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rule:
    key: str
    method: str
    path: str
    keywords: tuple[str, ...]  # all must appear in one retrieved doc chunk (case-insensitive)
    category: str
    title: str
    expected: str
    risk: int
    difficulty: float  # how hard the test is to write correctly (0..1)
    requests: tuple[tuple[str, str, tuple[str, ...], tuple[int, ...]], ...]
    code: str
    preconditions: tuple[str, ...] = field(default_factory=tuple)


ORDER = '{"items": [{"product_id": product_id, "quantity": 1}]}'
FIRST_PRODUCT = 'product_id = next(p["id"] for p in client.get("/products", headers=auth["customer"]).json() if p["active"])'

RULES: list[Rule] = [
    Rule(
        "qty_range",
        "POST",
        "/orders",
        ("quantity", "1 to 1000"),
        "validation",
        "Reject item quantities outside 1..1000",
        "422 for quantity 0, -1 and 1001; nothing is created",
        4,
        0.3,
        (("GET", "/products", (), (200,)), ("POST", "/orders", ("items",), (422,))),
        f"""
def TEST_NAME(client, auth):
    {FIRST_PRODUCT}
    for qty in (0, -1, 1001):
        r = client.post("/orders", json={{"items": [{{"product_id": product_id, "quantity": qty}}]}}, headers=auth["customer"])
        assert r.status_code == 422, f"quantity {{qty}} was accepted with {{r.status_code}}"
""",
    ),
    Rule(
        "half_up",
        "POST",
        "/orders",
        ("half-up", "line total"),
        "money/precision",
        "Line totals are unit price x quantity rounded half-up to the cent",
        "line_total equals Decimal(unit_price) * quantity quantized half-up; total is the sum",
        5,
        0.75,
        (("GET", "/products", (), (200,)), ("POST", "/orders", ("items",), (201,))),
        """
def TEST_NAME(client, auth):
    products = [p for p in client.get("/products", headers=auth["customer"]).json() if p["active"]]
    for p in products:
        for qty in (1, 3):
            r = client.post("/orders", json={"items": [{"product_id": p["id"], "quantity": qty}]}, headers=auth["customer"])
            assert r.status_code == 201, r.text
            expected = (Decimal(p["price"]) * qty).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            line = r.json()["items"][0]
            assert Decimal(line["line_total"]) == expected, f"{p['sku']} x{qty}: {line['line_total']} != {expected}"
            assert Decimal(r.json()["total"]) == expected
""",
    ),
    Rule(
        "idempotency_key",
        "POST",
        "/orders",
        ("idempotency-key", "original"),
        "idempotency",
        "Repeating a request with the same Idempotency-Key returns the original order",
        "first call 201; second call with the same key 200 with the same order id",
        4,
        0.4,
        (("GET", "/products", (), (200,)), ("POST", "/orders", ("items",), (201, 200))),
        f"""
def TEST_NAME(client, auth):
    {FIRST_PRODUCT}
    headers = {{**auth["customer"], "Idempotency-Key": f"agentqa-{{uuid.uuid4()}}"}}
    first = client.post("/orders", json={ORDER}, headers=headers)
    second = client.post("/orders", json={ORDER}, headers=headers)
    assert first.status_code == 201, first.text
    assert second.status_code == 200, f"replay returned {{second.status_code}}"
    assert second.json()["id"] == first.json()["id"], "a duplicate order was created"
""",
    ),
    Rule(
        "tz_offset",
        "GET",
        "/orders",
        ("created_from", "offset"),
        "time/timezone",
        "created_from converts offset timestamps to UTC",
        "an order created at T is returned for created_from = T-30min expressed in +02:00",
        3,
        0.8,
        (
            ("GET", "/products", (), (200,)),
            ("POST", "/orders", ("items",), (201,)),
            ("GET", "/orders", ("created_from", "page_size"), (200,)),
        ),
        f"""
def TEST_NAME(client, auth):
    {FIRST_PRODUCT}
    order = client.post("/orders", json={ORDER}, headers=auth["customer"]).json()
    created = datetime.datetime.fromisoformat(order["created_at"].replace("Z", "+00:00"))
    since = (created - timedelta(minutes=30)).astimezone(timezone(timedelta(hours=2))).isoformat()
    r = client.get("/orders", params={{"created_from": since, "page_size": 100}}, headers=auth["customer"])
    assert r.status_code == 200, r.text
    assert order["id"] in [o["id"] for o in r.json()["items"]], f"order missing for created_from={{since}}"
""",
    ),
    Rule(
        "list_own",
        "GET",
        "/orders",
        ("customers only ever see their own",),
        "auth/permission",
        "Customers only see their own orders in the list",
        "every listed order belongs to the calling customer",
        4,
        0.3,
        (("GET", "/orders", ("page_size",), (200,)),),
        """
def TEST_NAME(client, auth, customer_id):
    r = client.get("/orders", params={"page_size": 100}, headers=auth["customer"])
    assert r.status_code == 200
    assert {o["customer_id"] for o in r.json()["items"]} <= {customer_id}
""",
    ),
    Rule(
        "idor_order",
        "GET",
        "/orders/{order_id}",
        ("someone else's order", "404"),
        "auth/permission",
        "A customer cannot read another customer's order",
        "404 when a customer requests another customer's order id",
        5,
        0.5,
        (("GET", "/orders", (), (200,)), ("GET", "/orders/{order_id}", (), (404,))),
        """
def TEST_NAME(client, auth, find_id, other_customer_id):
    ids = find_id("/orders", auth["admin"], predicate=lambda o: o["customer_id"] == other_customer_id)
    if not ids:
        pytest.skip("no order owned by another customer")
    r = client.get(f"/orders/{ids[0]}", headers=auth["customer"])
    assert r.status_code == 404, f"customer read another customer's order: {r.status_code}"
""",
    ),
    Rule(
        "price_admin_only",
        "PATCH",
        "/products/{product_id}",
        ("admin only", "403"),
        "auth/permission",
        "Only admins can change product prices",
        "403 for staff and customer tokens; price unchanged",
        5,
        0.4,
        (("GET", "/products", (), (200,)), ("PATCH", "/products/{product_id}", ("price",), (403,))),
        """
def TEST_NAME(client, auth):
    product = client.get("/products", headers=auth["admin"]).json()[0]
    for role in ("staff", "customer"):
        r = client.patch(f"/products/{product['id']}", json={"price": "0.01"}, headers=auth[role])
        assert r.status_code == 403, f"{role} changed a price: {r.status_code}"
""",
    ),
    Rule(
        "ship_cancelled",
        "POST",
        "/orders/{order_id}/ship",
        ("only from `paid`", "cancelled"),
        "state-transition",
        "A cancelled order cannot be shipped",
        "409 when shipping an order that was cancelled",
        5,
        0.6,
        (
            ("GET", "/products", (), (200,)),
            ("POST", "/orders", ("items",), (201,)),
            ("POST", "/orders/{order_id}/cancel", (), (200,)),
            ("POST", "/orders/{order_id}/ship", (), (409,)),
        ),
        f"""
def TEST_NAME(client, auth):
    {FIRST_PRODUCT}
    order = client.post("/orders", json={ORDER}, headers=auth["customer"]).json()
    assert client.post(f"/orders/{{order['id']}}/cancel", headers=auth["customer"]).status_code == 200
    r = client.post(f"/orders/{{order['id']}}/ship", headers=auth["staff"])
    assert r.status_code == 409, f"cancelled order shipped: {{r.status_code}}"
""",
    ),
    Rule(
        "ship_unpaid",
        "POST",
        "/orders/{order_id}/ship",
        ("only from `paid`",),
        "state-transition",
        "An unpaid (pending) order cannot be shipped",
        "409 when shipping a pending order",
        4,
        0.5,
        (
            ("GET", "/products", (), (200,)),
            ("POST", "/orders", ("items",), (201,)),
            ("POST", "/orders/{order_id}/ship", (), (409,)),
        ),
        f"""
def TEST_NAME(client, auth):
    {FIRST_PRODUCT}
    order = client.post("/orders", json={ORDER}, headers=auth["customer"]).json()
    r = client.post(f"/orders/{{order['id']}}/ship", headers=auth["staff"])
    assert r.status_code == 409, f"pending order shipped: {{r.status_code}}"
""",
    ),
    Rule(
        "webhook_bad_sig",
        "POST",
        "/webhooks/payment",
        ("hmac-sha256", "401"),
        "webhooks/signatures",
        "Payment webhook rejects an invalid signature",
        "401 for a wrong X-Signature; the order stays pending",
        5,
        0.7,
        (
            ("GET", "/products", (), (200,)),
            ("POST", "/orders", ("items",), (201,)),
            ("POST", "/webhooks/payment", ("event_id", "order_id", "amount", "status"), (401,)),
            ("GET", "/orders/{order_id}", (), (200,)),
        ),
        f"""
def TEST_NAME(client, auth, webhook_header):
    {FIRST_PRODUCT}
    order = client.post("/orders", json={ORDER}, headers=auth["customer"]).json()
    body = json.dumps({{"event_id": f"evt-{{uuid.uuid4()}}", "order_id": order["id"], "amount": order["total"], "status": "succeeded"}}).encode()
    r = client.post("/webhooks/payment", content=body, headers={{webhook_header: "0" * 64, "Content-Type": "application/json"}})
    assert r.status_code == 401, f"unsigned payment accepted: {{r.status_code}}"
    assert client.get(f"/orders/{{order['id']}}", headers=auth["customer"]).json()["status"] == "pending"
""",
    ),
    Rule(
        "webhook_duplicate",
        "POST",
        "/webhooks/payment",
        ("repeated `event_id`",),
        "idempotency",
        "A repeated webhook event_id is acknowledged and ignored",
        "second delivery returns 200 and does not issue a second invoice",
        3,
        0.6,
        (
            ("GET", "/products", (), (200,)),
            ("POST", "/orders", ("items",), (201,)),
            ("POST", "/webhooks/payment", ("event_id", "order_id", "amount", "status"), (200,)),
        ),
        f"""
def TEST_NAME(client, auth, sign_webhook, webhook_header):
    {FIRST_PRODUCT}
    order = client.post("/orders", json={ORDER}, headers=auth["customer"]).json()
    body = json.dumps({{"event_id": f"evt-{{uuid.uuid4()}}", "order_id": order["id"], "amount": order["total"], "status": "succeeded"}}).encode()
    headers = {{webhook_header: sign_webhook(body), "Content-Type": "application/json"}}
    first = client.post("/webhooks/payment", content=body, headers=headers)
    second = client.post("/webhooks/payment", content=body, headers=headers)
    assert first.status_code == 200 and second.status_code == 200
    assert second.json() != first.json(), "duplicate event was processed twice"
""",
    ),
    Rule(
        "webhook_amount",
        "POST",
        "/webhooks/payment",
        ("amount must match",),
        "money/precision",
        "A payment whose amount differs from the order total is rejected",
        "409 when amount != order total; order stays pending",
        4,
        0.6,
        (
            ("GET", "/products", (), (200,)),
            ("POST", "/orders", ("items",), (201,)),
            ("POST", "/webhooks/payment", ("event_id", "order_id", "amount", "status"), (409,)),
        ),
        f"""
def TEST_NAME(client, auth, sign_webhook, webhook_header):
    {FIRST_PRODUCT}
    order = client.post("/orders", json={ORDER}, headers=auth["customer"]).json()
    wrong = str(Decimal(order["total"]) + Decimal("1.00"))
    body = json.dumps({{"event_id": f"evt-{{uuid.uuid4()}}", "order_id": order["id"], "amount": wrong, "status": "succeeded"}}).encode()
    r = client.post("/webhooks/payment", content=body, headers={{webhook_header: sign_webhook(body), "Content-Type": "application/json"}})
    assert r.status_code == 409, f"mismatched amount accepted: {{r.status_code}}"
""",
    ),
    Rule(
        "customer_fields",
        "GET",
        "/customers/{customer_id}",
        ("only these fields",),
        "data-leak",
        "Customer record exposes only the documented fields",
        "response keys are exactly id, name, email, created_at",
        4,
        0.3,
        (("GET", "/customers/{customer_id}", (), (200,)),),
        """
def TEST_NAME(client, auth, customer_id):
    r = client.get(f"/customers/{customer_id}", headers=auth["customer"])
    assert r.status_code == 200, r.text
    assert set(r.json()) == {"id", "name", "email", "created_at"}, f"unexpected fields: {sorted(r.json())}"
""",
    ),
    Rule(
        "customer_other",
        "GET",
        "/customers/{customer_id}",
        ("only their own record",),
        "auth/permission",
        "A customer cannot read another customer's record",
        "404 for another customer's record",
        4,
        0.3,
        (("GET", "/customers/{customer_id}", (), (404,)),),
        """
def TEST_NAME(client, auth, other_customer_id):
    r = client.get(f"/customers/{other_customer_id}", headers=auth["customer"])
    assert r.status_code == 404, f"customer read another customer's record: {r.status_code}"
""",
    ),
    Rule(
        "cancel_shipped",
        "POST",
        "/orders/{order_id}/cancel",
        ("only from `pending` or `paid`",),
        "state-transition",
        "A shipped order cannot be cancelled",
        "409 when cancelling a shipped order",
        3,
        0.5,
        (
            ("GET", "/orders", ("status",), (200,)),
            ("POST", "/orders/{order_id}/cancel", (), (409,)),
        ),
        """
def TEST_NAME(client, auth, find_id):
    ids = find_id("/orders?status=shipped", auth["admin"])
    if not ids:
        pytest.skip("no shipped order available")
    r = client.post(f"/orders/{ids[0]}/cancel", headers=auth["admin"])
    assert r.status_code == 409, f"shipped order cancelled: {r.status_code}"
""",
    ),
]
RULES_BY_KEY = {r.key: r for r in RULES}
RULES_BY_TITLE = {r.title.lower(): r for r in RULES}
