"""Deterministic seed data. IDs are uuid5 values so tests and docs can reference them."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal

from target_api.app import bugs
from target_api.app.db import Database, execute, query

NS = uuid.UUID("6f1c2a7e-0d4b-4f1e-9a51-3c2b8e7d9f10")


def sid(name: str) -> str:
    return str(uuid.uuid5(NS, name))


TOKENS: dict[str, dict[str, str | None]] = {
    "tok-admin": {"role": "admin", "customer_id": None},
    "tok-staff": {"role": "staff", "customer_id": None},
    "tok-alice": {"role": "customer", "customer_id": sid("customer:alice")},
    "tok-bob": {"role": "customer", "customer_id": sid("customer:bob")},
}

CUSTOMERS = [
    ("alice", "Alice Archer", "alice@example.test", 0.12),
    ("bob", "Bob Baker", "bob@example.test", 0.71),
    ("carol", "Carol Chen", "carol@example.test", 0.33),
]
PRODUCTS = [
    ("WIDGET", "Widget", "19.99", True),
    ("GADGET", "Gadget", "5.10", True),
    ("BOLT", "Bolt (each)", "0.125", True),
    ("CABLE", "Cable (per metre)", "2.345", True),
    ("NUT", "Nut (each)", "0.333", True),
    ("LEGACY", "Legacy part", "99.00", False),
]
EPOCH = datetime(2026, 3, 1, 9, 0, 0, tzinfo=UTC)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def line_total(unit_price: str, quantity: int) -> str:
    if bugs.on("B03"):
        return f"{round(float(unit_price) * quantity, 2):.2f}"
    return str((Decimal(unit_price) * quantity).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def insert_order(
    conn: sqlite3.Connection,
    order_id: str,
    customer_id: str,
    status: str,
    created_at: str,
    items: list[tuple[str, str, int]],
    idem: str | None = None,
) -> str:
    """items: (product_id, unit_price, quantity). Returns the order total."""
    totals = [line_total(price, qty) for _, price, qty in items]
    total = str(sum((Decimal(t) for t in totals), Decimal("0.00")))
    execute(
        conn,
        "INSERT INTO orders (id, customer_id, status, total, currency, created_at, idempotency_key)"
        " VALUES (?, ?, ?, ?, 'USD', ?, ?)",
        (order_id, customer_id, status, total, created_at, idem),
    )
    for (pid, price, qty), lt in zip(items, totals, strict=True):
        execute(
            conn,
            "INSERT INTO order_items (order_id, product_id, quantity, unit_price, line_total)"
            " VALUES (?, ?, ?, ?, ?)",
            (order_id, pid, qty, price, lt),
        )
    return total


def seed(db: Database) -> None:
    with db.write_lock, db.connection() as conn:
        for table in (
            "order_items",
            "invoices",
            "orders",
            "products",
            "customers",
            "webhook_events",
        ):
            execute(conn, f"DELETE FROM {table}")
        for key, name, email, risk in CUSTOMERS:
            execute(
                conn,
                "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
                (sid(f"customer:{key}"), name, email, iso(EPOCH - timedelta(days=60)), risk),
            )
        for sku, name, price, active in PRODUCTS:
            execute(
                conn,
                "INSERT INTO products VALUES (?, ?, ?, ?, 'USD', ?)",
                (sid(f"product:{sku}"), sku, name, price, int(active)),
            )
        widget, gadget = sid("product:WIDGET"), sid("product:GADGET")
        alice, bob, carol = sid("customer:alice"), sid("customer:bob"), sid("customer:carol")
        fixed = [
            ("order:alice-pending", alice, "pending", 0, [(widget, "19.99", 1)]),
            ("order:alice-paid", alice, "paid", 1, [(gadget, "5.10", 2)]),
            ("order:bob-pending", bob, "pending", 2, [(widget, "19.99", 3)]),
            ("order:bob-cancelled", bob, "cancelled", 3, [(gadget, "5.10", 1)]),
            ("order:alice-shipped", alice, "shipped", 4, [(widget, "19.99", 2)]),
        ]
        for name, cust, status, day, items in fixed:
            total = insert_order(
                conn, sid(name), cust, status, iso(EPOCH + timedelta(days=day, hours=day)), items
            )
            if status in ("paid", "shipped"):
                execute(
                    conn,
                    "INSERT INTO invoices VALUES (?, ?, ?, ?, ?)",
                    (
                        sid(f"invoice:{name}"),
                        sid(name),
                        total,
                        "paid",
                        iso(EPOCH + timedelta(days=day, hours=day + 1)),
                    ),
                )
        # 25 older pending orders for Carol, one per hour, to make pagination meaningful.
        for i in range(25):
            insert_order(
                conn,
                sid(f"order:carol-{i}"),
                carol,
                "pending",
                iso(EPOCH - timedelta(days=30) + timedelta(hours=i)),
                [(gadget, "5.10", 1)],
            )
        conn.commit()


def bulk_orders(db: Database, n: int) -> int:
    """Add ``n`` synthetic orders (for performance tests), spread over 2025."""
    customers = [sid(f"customer:{c[0]}") for c in CUSTOMERS]
    gadget = sid("product:GADGET")
    statuses = ["pending", "paid", "cancelled", "shipped"]
    start = datetime(2025, 1, 1, tzinfo=UTC)
    with db.write_lock, db.connection() as conn:
        existing = query(conn, "SELECT COUNT(*) AS c FROM orders")[0]["c"]
        orders, items = [], []
        for i in range(n):
            k = existing + i
            oid = sid(f"order:bulk-{k}")
            created = iso(start + timedelta(minutes=(k * 37) % 525_600))
            orders.append((oid, customers[k % 3], statuses[k % 4], "5.10", created))
            items.append((oid, gadget, 1, "5.10", "5.10"))
        conn.executemany(
            "INSERT INTO orders (id, customer_id, status, total, currency, created_at)"
            " VALUES (?, ?, ?, ?, 'USD', ?)",
            orders,
        )
        conn.executemany(
            "INSERT INTO order_items (order_id, product_id, quantity, unit_price, line_total)"
            " VALUES (?, ?, ?, ?, ?)",
            items,
        )
        conn.commit()
        total: int = query(conn, "SELECT COUNT(*) AS c FROM orders")[0]["c"]
    return total
