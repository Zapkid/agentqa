"""One SQLite database for run history, checkpoints, the delegation ledger, learned routing
stats, incremental-run artifacts and cross-run lessons. Plain SQL, no ORM."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from agentqa.config import agentqa_home

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, created REAL, spec_id TEXT, api_title TEXT, profile TEXT, strategy TEXT,
  status TEXT, trace_id TEXT, run_dir TEXT, summary TEXT
);
CREATE TABLE IF NOT EXISTS checkpoints (
  run_id TEXT, task_id TEXT, kind TEXT, status TEXT, payload TEXT, updated REAL,
  PRIMARY KEY (run_id, task_id)
);
CREATE TABLE IF NOT EXISTS delegations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, task_id TEXT, task_type TEXT, features TEXT,
  start_tier TEXT, reason TEXT, est_tokens INTEGER, actual_tokens INTEGER, cost_usd REAL,
  verifier_outcome TEXT, escalation_path TEXT, final_result TEXT, ts REAL
);
CREATE TABLE IF NOT EXISTS routing_stats (
  task_type TEXT, tier TEXT, successes INTEGER DEFAULT 0, failures INTEGER DEFAULT 0,
  tokens INTEGER DEFAULT 0, latency_s REAL DEFAULT 0, PRIMARY KEY (task_type, tier)
);
CREATE TABLE IF NOT EXISTS artifacts (
  api_key TEXT, endpoint TEXT, endpoint_hash TEXT, kind TEXT, payload TEXT, updated REAL,
  PRIMARY KEY (api_key, endpoint, kind)
);
CREATE TABLE IF NOT EXISTS lessons (
  id INTEGER PRIMARY KEY AUTOINCREMENT, api_key TEXT, endpoint TEXT, kind TEXT, text TEXT,
  source_run TEXT, created REAL, uses INTEGER DEFAULT 0, UNIQUE (api_key, endpoint, text)
);
"""


class Store:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (agentqa_home() / "agentqa.db")
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._lock:
            self.conn.execute(sql, params)
            self.conn.commit()

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(sql, params).fetchall())

    # -------------------------------------------------------------- checkpoints

    def checkpoint(self, run_id: str, task_id: str, kind: str, status: str, payload: Any) -> None:
        self.execute(
            "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, task_id, kind, status, json.dumps(payload, default=str), time.time()),
        )

    def checkpoints(self, run_id: str) -> dict[str, tuple[str, str, Any]]:
        rows = self.query(
            "SELECT task_id, kind, status, payload FROM checkpoints WHERE run_id = ?", (run_id,)
        )
        return {r["task_id"]: (r["kind"], r["status"], json.loads(r["payload"])) for r in rows}

    # -------------------------------------------------------------- runs

    # Columns save_run may set. Column names are interpolated into SQL, so they come from this
    # allowlist and never from a caller's keyword names.
    RUN_FIELDS = frozenset(
        {"spec_id", "api_title", "profile", "strategy", "status", "trace_id", "run_dir", "summary"}
    )

    def save_run(self, run_id: str, **fields: Any) -> None:
        unknown = set(fields) - self.RUN_FIELDS
        if unknown:
            raise ValueError(f"unknown run field(s): {sorted(unknown)}")
        existing = self.query("SELECT run_id FROM runs WHERE run_id = ?", (run_id,))
        if not existing:
            self.execute("INSERT INTO runs (run_id, created) VALUES (?, ?)", (run_id, time.time()))
        for k, v in fields.items():
            self.execute(
                f"UPDATE runs SET {k} = ? WHERE run_id = ?",  # noqa: S608 - k is allowlisted above
                (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v, run_id),
            )

    def runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.query("SELECT * FROM runs ORDER BY created DESC LIMIT ?", (limit,))
        ]

    def run(self, run_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        return dict(rows[0]) if rows else None
