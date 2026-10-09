"""Process-wide state shared by the routes: database, secrets, logging and per-route stats."""

import logging
import os
import threading
import time

from opentelemetry import trace

from target_api.app.db import Database
from target_api.app.seed import seed

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

db = Database()
seed(db)

_route_stats: dict[str, dict[str, float]] = {}
_stats_lock = threading.Lock()
_leak_cache: list[tuple[str, bytes]] = []  # P06
_cpu_start = time.process_time()
_wall_start = time.monotonic()
