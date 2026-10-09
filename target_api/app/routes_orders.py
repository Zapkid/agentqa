"""Orders, cancellation, shipping and invoices."""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Query, Response

from target_api.app import bugs
from target_api.app.db import execute, query
from target_api.app.deps import (
    Auth,
    UuidPath,
    _orders_json,
    _parse_created_from,
    _visible_order,
    parse_uuid,
    require,
)
from target_api.app.schemas import Invoice, Order, OrderIn, OrderPage, OrderStatus, errs
from target_api.app.seed import insert_order, iso
from target_api.app.state import db

router = APIRouter()


@router.post(
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


@router.get("/orders", response_model=OrderPage, tags=["orders"], responses=errs(401, 422))
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
        total = query(conn, f"SELECT COUNT(*) AS c FROM orders{clause}", params)[0]["c"]  # noqa: S608 - clause is assembled from literals; values are bound
        rows = query(
            conn,
            f"SELECT * FROM orders{clause} ORDER BY created_at DESC, id LIMIT ? OFFSET ?",  # noqa: S608 - clause is assembled from literals; values are bound
            [*params, page_size, offset],
        )
        return {
            "items": _orders_json(conn, rows),
            "page": page,
            "page_size": page_size,
            "total": total,
        }


@router.get(
    "/orders/{order_id}", response_model=Order, tags=["orders"], responses=errs(401, 404, 422)
)
def get_order(order_id: UuidPath, p: Auth) -> Any:
    oid = parse_uuid(order_id, "order_id")
    with db.connection() as conn:
        row = _visible_order(conn, p, oid)
        return _orders_json(conn, [row])[0]


@router.post(
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


@router.post(
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


@router.get(
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
