"""Test-environment endpoints (hidden from the OpenAPI document; this sandbox target only)."""

import os
import resource
import time
from typing import Any

from fastapi import APIRouter

from target_api.app import bugs
from target_api.app.seed import bulk_orders, seed
from target_api.app.state import _cpu_start, _leak_cache, _route_stats, _stats_lock, _wall_start, db

router = APIRouter()


@router.post("/__seed", include_in_schema=False)
def seed_more(orders: int = 0) -> dict[str, int]:
    return {"orders": bulk_orders(db, min(orders, 500_000))}


@router.post("/__reset", include_in_schema=False)
def reset() -> dict[str, str]:
    seed(db)
    _leak_cache.clear()
    with _stats_lock:
        _route_stats.clear()
    return {"status": "reset"}


@router.get("/__metrics", include_in_schema=False)
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
