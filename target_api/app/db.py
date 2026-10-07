"""SQLite storage with a small connection pool, query spans and per-request stats.

Every query goes through ``query()`` so the target can report, per request, how many queries
ran and how long the database took. That is what makes N+1 and missing-index defects visible
in traces and in ``/__metrics``.
"""

from __future__ import annotations

import os
import queue
import sqlite3
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace

from target_api.app import bugs

tracer = trace.get_tracer("target_api.db")


@dataclass
class RequestStats:
    queries: int = 0
    db_ms: float = 0.0


request_stats: ContextVar[RequestStats | None] = ContextVar("request_stats", default=None)


class PoolExhausted(Exception):
    pass


SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL, created_at TEXT NOT NULL,
  internal_risk_score REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS products (
  id TEXT PRIMARY KEY, sku TEXT UNIQUE NOT NULL, name TEXT NOT NULL, price TEXT NOT NULL,
  currency TEXT NOT NULL DEFAULT 'USD', active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS orders (
  id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(id), status TEXT NOT NULL,
  total TEXT NOT NULL, currency TEXT NOT NULL DEFAULT 'USD', created_at TEXT NOT NULL,
  idempotency_key TEXT
);
CREATE TABLE IF NOT EXISTS order_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT NOT NULL REFERENCES orders(id),
  product_id TEXT NOT NULL, quantity INTEGER NOT NULL, unit_price TEXT NOT NULL,
  line_total TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS invoices (
  id TEXT PRIMARY KEY, order_id TEXT UNIQUE NOT NULL, amount TEXT NOT NULL, status TEXT NOT NULL,
  issued_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS webhook_events (event_id TEXT PRIMARY KEY, received_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_items_order ON order_items(order_id);
CREATE INDEX IF NOT EXISTS ix_orders_customer ON orders(customer_id);
"""
# Omitted when P02 is on (missing index defect).
FILTER_INDEXES = """
CREATE INDEX IF NOT EXISTS ix_orders_created ON orders(created_at);
CREATE INDEX IF NOT EXISTS ix_orders_status_created ON orders(status, created_at);
"""


class Database:
    def __init__(self, path: str | None = None) -> None:
        self.path = (
            path
            or os.environ.get("TARGET_DB")
            or os.path.join(tempfile.mkdtemp(prefix="target_api_"), "orders.db")
        )
        size = 1 if bugs.perf("P05") else 8
        self.acquire_timeout = 0.005 if bugs.perf("P05") else 5.0
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue()
        for _ in range(size):
            conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._pool.put(conn)
        self.write_lock = threading.Lock()
        with self.connection() as conn:
            conn.executescript(SCHEMA)
            if not bugs.perf("P02"):
                conn.executescript(FILTER_INDEXES)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        try:
            conn = self._pool.get(timeout=self.acquire_timeout)
        except queue.Empty as exc:
            raise PoolExhausted("database connection pool exhausted") from exc
        try:
            yield conn
        finally:
            self._pool.put(conn)


def query(
    conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] | list[Any] = ()
) -> list[sqlite3.Row]:
    stats = request_stats.get()
    with tracer.start_as_current_span("db.query") as span:
        span.set_attribute("db.system", "sqlite")
        span.set_attribute("db.statement", " ".join(sql.split())[:300])
        t0 = time.perf_counter()
        rows = conn.execute(sql, params).fetchall()
        elapsed = (time.perf_counter() - t0) * 1000
        span.set_attribute("db.duration_ms", round(elapsed, 3))
    if stats is not None:
        stats.queries += 1
        stats.db_ms += elapsed
    return rows


def execute(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> None:
    query(conn, sql, params)
