"""Orders and Invoicing API: the system under test.

A deliberately small FastAPI service with SQLite storage, token auth for three roles (admin,
staff, customer) and seeded defects toggled by ``BUGS`` / ``PERF_BUGS`` (see ``bugs.yaml`` and
``perf_bugs.yaml``). The OpenAPI document is identical for every build; only behaviour changes.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import resource
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from opentelemetry import trace
from pydantic import BaseModel, Field

from target_api.app import bugs
from target_api.app.db import Database, PoolExhausted, RequestStats, execute, query, request_stats
from target_api.app.seed import TOKENS, bulk_orders, insert_order, iso, seed

tracer = trace.get_tracer("target_api")
WEBHOOK_SECRET = os.environ.get("TARGET_WEBHOOK_SECRET", "change-me-local-only")
LOG_PATH = os.environ.get("TARGET_LOG")

_log = logging.getLogger("target_api")
if LOG_PATH:
    handler = logging.FileHandler(LOG_PATH)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(handler)
    _log.setLevel(logging.INFO)
    _log.propagate = False


def _setup_tracing() -> None:
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if os.environ.get("TARGET_TRACING", "0") != "1" or not endpoint:
        return
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "target-api"}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces"))
    )
    trace.set_tracer_provider(provider)


_setup_tracing()

app = FastAPI(
    title="Orders and Invoicing API",
    version="1.0.0",
    description="Customers, products, orders, invoices and a signed payment webhook. "
    "All endpoints except /health and /webhooks/payment require `Authorization: Bearer <token>`.",
)
db = Database()
seed(db)

# ------------------------------------------------------------------ schemas


class OrderStatus(StrEnum):
    pending = "pending"
    paid = "paid"
    cancelled = "cancelled"
    shipped = "shipped"


class Customer(BaseModel):
    id: str = Field(json_schema_extra={"format": "uuid"})
    name: str
    email: str
    created_at: datetime


class Product(BaseModel):
    id: str = Field(json_schema_extra={"format": "uuid"})
    sku: str
    name: str
    price: str = Field(
        description="Unit price as a decimal string with up to 3 decimals", examples=["19.99"]
    )
    currency: str
    active: bool


class ProductPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    price: str | None = Field(default=None, pattern=r"^\d{1,6}(\.\d{1,3})?$", examples=["21.50"])
    active: bool | None = None


class OrderItemIn(BaseModel):
    product_id: str = Field(json_schema_extra={"format": "uuid"})
    quantity: int = Field(description="Units ordered, 1..1000", examples=[2])


class OrderIn(BaseModel):
    customer_id: str | None = Field(
        default=None,
        json_schema_extra={"format": "uuid"},
        description="Required for staff/admin; customers always order for themselves",
    )
    items: list[OrderItemIn] = Field(min_length=1, max_length=50)


class OrderItem(BaseModel):
    product_id: str = Field(json_schema_extra={"format": "uuid"})
    quantity: int
    unit_price: str
    line_total: str = Field(description="unit_price * quantity rounded half-up to cents")


class Order(BaseModel):
    id: str = Field(json_schema_extra={"format": "uuid"})
    customer_id: str = Field(json_schema_extra={"format": "uuid"})
    status: OrderStatus
    items: list[OrderItem]
    total: str
    currency: str
    created_at: datetime


class OrderPage(BaseModel):
    items: list[Order]
    page: int
    page_size: int
    total: int


class Invoice(BaseModel):
    id: str = Field(json_schema_extra={"format": "uuid"})
    order_id: str = Field(json_schema_extra={"format": "uuid"})
    amount: str
    status: str
    issued_at: datetime


class PaymentEvent(BaseModel):
    event_id: str = Field(min_length=1, max_length=64)
    order_id: str = Field(json_schema_extra={"format": "uuid"})
    amount: str
    status: str = Field(description="'succeeded' or 'failed'")


class Error(BaseModel):
    detail: str


ERR: dict[int, dict[str, Any]] = {
    401: {"model": Error, "description": "Missing or invalid token"},
    403: {"model": Error, "description": "Role not allowed"},
    404: {"model": Error, "description": "Not found (or not visible to the caller)"},
    409: {"model": Error, "description": "Invalid state transition"},
    422: {"description": "Validation error"},
    503: {"model": Error, "description": "Database busy"},
}


def errs(*codes: int) -> dict[int | str, dict[str, Any]]:
    return {c: ERR[c] for c in codes}


# ------------------------------------------------------------------ middleware and stats

_route_stats: dict[str, dict[str, float]] = {}
_stats_lock = threading.Lock()
_leak_cache: list[tuple[str, bytes]] = []  # P06
_cpu_start = time.process_time()
_wall_start = time.monotonic()


@app.middleware("http")
async def instrument(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    stats = RequestStats()
    token = request_stats.set(stats)
    t0 = time.perf_counter()
    with tracer.start_as_current_span(f"HTTP {request.method}") as span:
        try:
            response = await call_next(request)
        except Exception as exc:  # unhandled errors become a logged 500
            if LOG_PATH:
                _log.error(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "level": "error",
                            "method": request.method,
                            "path": request.url.path,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                )
            response = JSONResponse({"detail": "internal server error"}, status_code=500)
        route = request.scope.get("route")
        template = getattr(route, "path", request.url.path)
        span.set_attribute("http.route", template)
        span.set_attribute("http.status_code", response.status_code)
        span.set_attribute("db.query_count", stats.queries)
    elapsed = (time.perf_counter() - t0) * 1000
    request_stats.reset(token)
    if bugs.perf("P06") and not template.startswith("/__"):
        _leak_cache.append((str(uuid.uuid4()), b"x" * 20_000))
        sum(1 for k, _ in _leak_cache if k == "")  # linear scan: latency drifts with size
    size = int(response.headers.get("content-length", 0) or 0)
    key = f"{request.method} {template}"
    with _stats_lock:
        row = _route_stats.setdefault(
            key,
            {
                "requests": 0,
                "errors": 0,
                "total_ms": 0.0,
                "db_ms": 0.0,
                "db_queries": 0,
                "response_bytes": 0,
                "max_ms": 0.0,
            },
        )
        row["requests"] += 1
        row["errors"] += int(response.status_code >= 500)
        row["total_ms"] += elapsed
        row["db_ms"] += stats.db_ms
        row["db_queries"] += stats.queries
        row["response_bytes"] += size
        row["max_ms"] = max(row["max_ms"], elapsed)
    if LOG_PATH:
        _log.info(
            json.dumps(
                {
                    "ts": time.time(),
                    "method": request.method,
                    "path": request.url.path,
                    "route": template,
                    "status": response.status_code,
                    "ms": round(elapsed, 2),
                    "db_queries": stats.queries,
                }
            )
        )
    return response


@app.exception_handler(PoolExhausted)
async def pool_exhausted(request: Request, exc: PoolExhausted) -> JSONResponse:
    return JSONResponse({"detail": "database busy"}, status_code=503)


# ------------------------------------------------------------------ auth


class Principal(BaseModel):
    role: str
    customer_id: str | None


bearer = HTTPBearer(auto_error=False, description="Opaque role token, e.g. `Bearer tok-admin`")


def principal(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)] = None,
) -> Principal:
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(401, "missing bearer token")
    info = TOKENS.get(creds.credentials.strip())
    if info is None:
        raise HTTPException(401, "invalid token")
    return Principal(role=str(info["role"]), customer_id=info["customer_id"])


Auth = Annotated[Principal, Depends(principal)]


def require(p: Principal, *roles: str) -> None:
    if p.role not in roles:
        raise HTTPException(403, f"role {p.role} not allowed")


def parse_uuid(value: str, what: str) -> str:
    if bugs.on("B08"):
        return str(uuid.UUID(value))  # raises ValueError -> 500
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise HTTPException(422, f"{what} must be a UUID") from exc


UuidPath = Annotated[str, Path(json_schema_extra={"format": "uuid"})]

# ------------------------------------------------------------------ helpers


def _customer_json(row: Any) -> dict[str, Any]:
    out = {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "created_at": row["created_at"],
    }
    if bugs.on("B10"):
        out["internal_risk_score"] = row["internal_risk_score"]
    return out


def _product_json(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "sku": row["sku"],
        "name": row["name"],
        "price": row["price"],
        "currency": row["currency"],
        "active": bool(row["active"]),
    }


def _orders_json(conn: Any, rows: list[Any]) -> list[dict[str, Any]]:
    if not rows:
        return []
    if bugs.perf("P01"):
        out = []
        for r in rows:  # N+1: one customer query and one items query per order
            query(conn, "SELECT id FROM customers WHERE id = ?", (r["customer_id"],))
            items = query(
                conn, "SELECT * FROM order_items WHERE order_id = ? ORDER BY id", (r["id"],)
            )
            out.append(_order_dict(r, items))
        return out
    ids = [r["id"] for r in rows]
    marks = ",".join("?" * len(ids))
    items = query(conn, f"SELECT * FROM order_items WHERE order_id IN ({marks}) ORDER BY id", ids)
    by_order: dict[str, list[Any]] = {}
    for it in items:
        by_order.setdefault(it["order_id"], []).append(it)
    return [_order_dict(r, by_order.get(r["id"], [])) for r in rows]


def _order_dict(row: Any, items: list[Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "customer_id": row["customer_id"],
        "status": row["status"],
        "items": [
            {
                "product_id": i["product_id"],
                "quantity": i["quantity"],
                "unit_price": i["unit_price"],
                "line_total": i["line_total"],
            }
            for i in items
        ],
        "total": row["total"],
        "currency": row["currency"],
        "created_at": row["created_at"],
    }


def _visible_order(conn: Any, p: Principal, order_id: str) -> Any:
    rows = query(conn, "SELECT * FROM orders WHERE id = ?", (order_id,))
    if not rows:
        raise HTTPException(404, "order not found")
    row = rows[0]
    if p.role == "customer" and row["customer_id"] != p.customer_id and not bugs.on("B04"):
        raise HTTPException(404, "order not found")
    return row


def _parse_created_from(value: str) -> str:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(422, "created_from must be an ISO 8601 timestamp") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)  # documented: naive timestamps are UTC
    elif bugs.on("B12"):
        dt = dt.replace(tzinfo=UTC)  # bug: keeps wall-clock time, drops the offset
    return iso(dt)


# ------------------------------------------------------------------ endpoints


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/customers", response_model=list[Customer], tags=["customers"], responses=errs(401, 403))
def list_customers(p: Auth) -> Any:
    require(p, "admin", "staff")
    with db.connection() as conn:
        return [_customer_json(r) for r in query(conn, "SELECT * FROM customers ORDER BY name")]


@app.get(
    "/customers/{customer_id}",
    response_model=Customer,
    tags=["customers"],
    responses=errs(401, 403, 404, 422),
)
def get_customer(customer_id: UuidPath, p: Auth) -> Any:
    cid = parse_uuid(customer_id, "customer_id")
    if p.role == "customer" and p.customer_id != cid:
        raise HTTPException(404, "customer not found")
    with db.connection() as conn:
        rows = query(conn, "SELECT * FROM customers WHERE id = ?", (cid,))
    if not rows:
        raise HTTPException(404, "customer not found")
    return JSONResponse(_customer_json(rows[0]))  # JSONResponse: bypasses response_model filtering


@app.get("/products", response_model=list[Product], tags=["products"], responses=errs(401))
def list_products(p: Auth) -> Any:
    with db.connection() as conn:
        return [_product_json(r) for r in query(conn, "SELECT * FROM products ORDER BY sku")]


@app.get(
    "/products/{product_id}",
    response_model=Product,
    tags=["products"],
    responses=errs(401, 404, 422),
)
def get_product(product_id: UuidPath, p: Auth) -> Any:
    pid = parse_uuid(product_id, "product_id")
    with db.connection() as conn:
        rows = query(conn, "SELECT * FROM products WHERE id = ?", (pid,))
    if not rows:
        raise HTTPException(404, "product not found")
    return _product_json(rows[0])


@app.patch(
    "/products/{product_id}",
    response_model=Product,
    tags=["products"],
    responses=errs(401, 403, 404, 422),
)
def patch_product(product_id: UuidPath, body: ProductPatch, p: Auth) -> Any:
    """Update a product. Admin only."""
    if not bugs.on("B05"):
        require(p, "admin")
    pid = parse_uuid(product_id, "product_id")
    with db.write_lock, db.connection() as conn:
        rows = query(conn, "SELECT * FROM products WHERE id = ?", (pid,))
        if not rows:
            raise HTTPException(404, "product not found")
        for field in ("name", "price", "active"):
            value = getattr(body, field)
            if value is not None:
                execute(
                    conn,
                    f"UPDATE products SET {field} = ? WHERE id = ?",
                    (int(value) if field == "active" else value, pid),
                )
        conn.commit()
        return _product_json(query(conn, "SELECT * FROM products WHERE id = ?", (pid,))[0])


@app.post(
    "/orders",
    response_model=Order,
    status_code=201,
    tags=["orders"],
    responses={
        200: {
            "model": Order,
            "description": "Replay of an earlier request with the same Idempotency-Key",
        },
        **errs(401, 403, 404, 422),
    },
)
def create_order(
    body: OrderIn,
    p: Auth,
    response: Response,
    idempotency_key: Annotated[str | None, Header(max_length=64)] = None,
) -> Any:
    """Create an order. Customers order for themselves; staff/admin must pass customer_id.

    Repeating a request with the same `Idempotency-Key` returns the original order with 200."""
    if p.role == "customer":
        customer_id = p.customer_id
    else:
        if not body.customer_id:
            raise HTTPException(422, "customer_id is required for staff/admin")
        customer_id = parse_uuid(body.customer_id, "customer_id")
    for item in body.items:
        if (item.quantity < 1 and not bugs.on("B01")) or item.quantity > 1000:
            raise HTTPException(422, "quantity must be between 1 and 1000")
    with db.write_lock, db.connection() as conn:
        if idempotency_key and not bugs.on("B06"):
            prior = query(
                conn,
                "SELECT * FROM orders WHERE idempotency_key = ? AND customer_id = ?",
                (idempotency_key, customer_id),
            )
            if prior:
                response.status_code = 200
                return _orders_json(conn, prior)[0]
        if not query(conn, "SELECT id FROM customers WHERE id = ?", (customer_id,)):
            raise HTTPException(404, "customer not found")
        lines = []
        for item in body.items:
            pid = parse_uuid(item.product_id, "product_id")
            prod = query(conn, "SELECT * FROM products WHERE id = ? AND active = 1", (pid,))
            if not prod:
                raise HTTPException(422, f"product {pid} not found or inactive")
            lines.append((pid, prod[0]["price"], item.quantity))
        order_id = str(uuid.uuid4())
        insert_order(
            conn,
            order_id,
            str(customer_id),
            "pending",
            iso(datetime.now(UTC)),
            lines,
            idempotency_key,
        )
        conn.commit()
        row = query(conn, "SELECT * FROM orders WHERE id = ?", (order_id,))
        return _orders_json(conn, row)[0]


@app.get("/orders", response_model=OrderPage, tags=["orders"], responses=errs(401, 422))
def list_orders(
    p: Auth,
    page: Annotated[int, Query(ge=1, description="1-based page number")] = 1,
    page_size: Annotated[int, Query(ge=1, description="Items per page, max 100")] = 20,
    status: Annotated[
        str | None,
        Query(
            description="Filter by status",
            json_schema_extra={"enum": [s.value for s in OrderStatus]},
        ),
    ] = None,
    created_from: Annotated[
        str | None,
        Query(
            description="ISO 8601; naive values are UTC", json_schema_extra={"format": "date-time"}
        ),
    ] = None,
) -> Any:
    """List orders, newest first. Customers see only their own orders."""
    if page_size > 100 and not bugs.perf("P03"):
        raise HTTPException(422, "page_size must be <= 100")
    where, params = [], []
    if p.role == "customer":
        where.append("customer_id = ?")
        params.append(p.customer_id)
    if status is not None:
        if status in {s.value for s in OrderStatus}:
            where.append("status = ?")
            params.append(status)
        elif not bugs.on("B09"):
            raise HTTPException(422, f"unknown status {status!r}")
    if created_from is not None:
        where.append("created_at >= ?")
        params.append(_parse_created_from(created_from))
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    offset = (page - 1) * page_size + (1 if bugs.on("B02") and page > 1 else 0)
    with db.connection() as conn:
        total = query(conn, f"SELECT COUNT(*) AS c FROM orders{clause}", params)[0]["c"]
        rows = query(
            conn,
            f"SELECT * FROM orders{clause} ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
            [*params, page_size, offset],
        )
        return {
            "items": _orders_json(conn, rows),
            "page": page,
            "page_size": page_size,
            "total": total,
        }


@app.get("/orders/{order_id}", response_model=Order, tags=["orders"], responses=errs(401, 404, 422))
def get_order(order_id: UuidPath, p: Auth) -> Any:
    oid = parse_uuid(order_id, "order_id")
    with db.connection() as conn:
        row = _visible_order(conn, p, oid)
        return _orders_json(conn, [row])[0]


@app.post(
    "/orders/{order_id}/cancel",
    response_model=Order,
    tags=["orders"],
    responses=errs(401, 404, 409, 422),
)
def cancel_order(order_id: UuidPath, p: Auth) -> Any:
    """Cancel a pending or paid order. Owner, staff or admin."""
    oid = parse_uuid(order_id, "order_id")
    with db.write_lock, db.connection() as conn:
        row = _visible_order(conn, p, oid)
        if row["status"] not in ("pending", "paid"):
            raise HTTPException(409, f"cannot cancel an order in status {row['status']}")
        execute(conn, "UPDATE orders SET status = 'cancelled' WHERE id = ?", (oid,))
        conn.commit()
        return _orders_json(conn, query(conn, "SELECT * FROM orders WHERE id = ?", (oid,)))[0]


@app.post(
    "/orders/{order_id}/ship",
    response_model=Order,
    tags=["orders"],
    responses=errs(401, 403, 404, 409, 422),
)
def ship_order(order_id: UuidPath, p: Auth) -> Any:
    """Mark a paid order as shipped. Staff or admin."""
    require(p, "admin", "staff")
    oid = parse_uuid(order_id, "order_id")
    with db.write_lock, db.connection() as conn:
        row = _visible_order(conn, p, oid)
        allowed = ("paid", "cancelled") if bugs.on("B07") else ("paid",)
        if row["status"] not in allowed:
            raise HTTPException(409, f"cannot ship an order in status {row['status']}")
        execute(conn, "UPDATE orders SET status = 'shipped' WHERE id = ?", (oid,))
        conn.commit()
        return _orders_json(conn, query(conn, "SELECT * FROM orders WHERE id = ?", (oid,)))[0]


@app.get(
    "/orders/{order_id}/invoice",
    response_model=Invoice,
    tags=["invoices"],
    responses=errs(401, 404, 422),
)
def get_order_invoice(order_id: UuidPath, p: Auth) -> Any:
    oid = parse_uuid(order_id, "order_id")
    with db.connection() as conn:
        _visible_order(conn, p, oid)
        rows = query(conn, "SELECT * FROM invoices WHERE order_id = ?", (oid,))
    if not rows:
        raise HTTPException(404, "invoice not found")
    return dict(rows[0])


@app.post(
    "/webhooks/payment",
    tags=["webhooks"],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": PaymentEvent.model_json_schema()}},
        }
    },
    responses={
        200: {"description": "Processed (or duplicate event ignored)"},
        401: {"model": Error, "description": "Invalid signature"},
        404: {"model": Error, "description": "Unknown order"},
        409: {"model": Error, "description": "Order not payable"},
        422: {"description": "Validation error"},
    },
)
async def payment_webhook(
    request: Request,
    x_signature: Annotated[
        str | None, Header(description="hex HMAC-SHA256 of the raw body")
    ] = None,
) -> dict[str, str]:
    """Payment provider callback. Authenticated by `X-Signature` (HMAC-SHA256 of the raw body
    with the shared secret), not by a bearer token. Duplicate `event_id`s are ignored."""
    raw = await request.body()
    expected = hmac.new(WEBHOOK_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    if not bugs.on("B11") and not (x_signature and hmac.compare_digest(expected, x_signature)):
        raise HTTPException(401, "invalid signature")
    try:
        event = PaymentEvent.model_validate_json(raw)
    except ValueError as exc:
        raise HTTPException(422, "invalid payment event") from exc
    if bugs.perf("P04"):
        time.sleep(0.05)  # blocking call inside the event loop
    else:
        await asyncio.sleep(0.05)  # same latency, but non-blocking
    return await asyncio.to_thread(_apply_payment, event)


def _apply_payment(event: PaymentEvent) -> dict[str, str]:
    oid = parse_uuid(event.order_id, "order_id")
    with db.write_lock, db.connection() as conn:
        if query(conn, "SELECT 1 FROM webhook_events WHERE event_id = ?", (event.event_id,)):
            return {"result": "duplicate"}
        rows = query(conn, "SELECT * FROM orders WHERE id = ?", (oid,))
        if not rows:
            raise HTTPException(404, "order not found")
        execute(
            conn,
            "INSERT INTO webhook_events VALUES (?, ?)",
            (event.event_id, iso(datetime.now(UTC))),
        )
        if event.status != "succeeded":
            conn.commit()
            return {"result": "ignored"}
        if rows[0]["status"] != "pending":
            conn.commit()
            raise HTTPException(409, "order is not pending")
        if Decimal(event.amount) != Decimal(rows[0]["total"]):
            conn.commit()
            raise HTTPException(409, "amount does not match order total")
        execute(conn, "UPDATE orders SET status = 'paid' WHERE id = ?", (oid,))
        execute(
            conn,
            "INSERT INTO invoices VALUES (?, ?, ?, 'paid', ?)",
            (str(uuid.uuid4()), oid, rows[0]["total"], iso(datetime.now(UTC))),
        )
        conn.commit()
    return {"result": "paid"}


# ------------------------------------------------------------------ test-environment endpoints
# Hidden from the OpenAPI document: they exist only in this sandbox target.


@app.post("/__seed", include_in_schema=False)
def seed_more(orders: int = 0) -> dict[str, int]:
    return {"orders": bulk_orders(db, min(orders, 500_000))}


@app.post("/__reset", include_in_schema=False)
def reset() -> dict[str, str]:
    seed(db)
    _leak_cache.clear()
    with _stats_lock:
        _route_stats.clear()
    return {"status": "reset"}


@app.get("/__metrics", include_in_schema=False)
def target_metrics() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    wall = time.monotonic() - _wall_start
    rss_mb = _rss_mb()
    with _stats_lock:
        routes = {k: dict(v) for k, v in _route_stats.items()}
    return {
        "build": bugs.active(),
        "rss_mb": rss_mb,
        "max_rss_mb": round(usage.ru_maxrss / 1024, 1),
        "cpu_s": round(time.process_time() - _cpu_start, 3),
        "uptime_s": round(wall, 3),
        "leak_cache_entries": len(_leak_cache),
        "routes": routes,
    }


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 1_048_576, 1)
    except (OSError, ValueError):
        return 0.0
