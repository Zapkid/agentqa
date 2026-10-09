"""The signed payment webhook."""

import asyncio
import hashlib
import hmac
import time
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated

from fastapi import (
    APIRouter,
    Header,
    HTTPException,
    Request,
)

from target_api.app import bugs
from target_api.app.db import execute, query
from target_api.app.deps import parse_uuid
from target_api.app.schemas import Error, PaymentEvent
from target_api.app.seed import iso
from target_api.app.state import WEBHOOK_SECRET, db

router = APIRouter()


@router.post(
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
