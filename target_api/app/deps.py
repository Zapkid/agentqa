"""Authentication, path-parameter parsing and the JSON helpers the routes share."""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Path
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from target_api.app import bugs
from target_api.app.db import query
from target_api.app.seed import TOKENS, iso


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
    items = query(conn, f"SELECT * FROM order_items WHERE order_id IN ({marks}) ORDER BY id", ids)  # noqa: S608 - only ? placeholders are interpolated
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
