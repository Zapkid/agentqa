"""Hand-written measurements proving each seeded performance defect exists.

Each test measures the same quantity on the clean build and on the defect build, back to back
on the same host, and asserts a large relative difference (not an absolute threshold).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from target_api.launcher import TargetServer

pytestmark = pytest.mark.slow
ADMIN = {"Authorization": "Bearer tok-admin"}


def _route(metrics: dict[str, object], key: str) -> dict[str, float]:
    routes = metrics["routes"]
    assert isinstance(routes, dict)
    row: dict[str, float] = routes[key]
    return row


def _queries_per_list(perf_bugs: list[str]) -> float:
    with TargetServer(perf_bugs=perf_bugs) as t:
        for _ in range(5):
            httpx.get(f"{t.base_url}/orders", params={"page_size": 30}, headers=ADMIN)
        row = _route(t.metrics(), "GET /orders")
        return row["db_queries"] / row["requests"]


def test_p01_n_plus_one() -> None:
    clean, buggy = _queries_per_list([]), _queries_per_list(["P01"])
    assert clean <= 4
    assert buggy >= 10 * clean  # ~2 extra queries per order on the page


def _filter_server_ms(perf_bugs: list[str]) -> tuple[float, float]:
    """(server ms per request, DB share of server time) for a filtered list query."""
    with (
        TargetServer(perf_bugs=perf_bugs) as t,
        httpx.Client(base_url=t.base_url, headers=ADMIN, timeout=30) as c,
    ):
        t.seed(150_000)
        for _ in range(7):
            r = c.get(
                "/orders",
                params={"status": "paid", "created_from": "2025-11-01T00:00:00Z", "page_size": 20},
            )
            assert r.status_code == 200
        row = _route(t.metrics(), "GET /orders")
        return row["total_ms"] / row["requests"], row["db_ms"] / row["total_ms"]


def test_p02_missing_index() -> None:
    (clean_ms, _), (buggy_ms, buggy_db_share) = _filter_server_ms([]), _filter_server_ms(["P02"])
    assert buggy_ms > 3 * clean_ms, (clean_ms, buggy_ms)
    assert buggy_db_share > 0.6  # DB time dominates the request


def test_p03_unbounded_payload() -> None:
    for perf_bugs, expect in (([], 422), (["P03"], 200)):
        with TargetServer(perf_bugs=perf_bugs) as t:
            t.seed(3000)
            r = httpx.get(
                f"{t.base_url}/orders", params={"page_size": 3000}, headers=ADMIN, timeout=30
            )
            assert r.status_code == expect
            if expect == 200:
                assert len(r.content) > 500_000


def _webhook_burst_s(perf_bugs: list[str]) -> float:
    limits = httpx.Limits(max_connections=40)
    with (
        TargetServer(perf_bugs=perf_bugs) as t,
        httpx.Client(base_url=t.base_url, limits=limits, timeout=30) as c,
    ):

        def call(i: int) -> int:
            # Invalid signature is rejected before the slow call, so sign properly via a bad
            # order id: the slow call happens, then 404. Throughput is what matters here.
            import json

            from target_api.tests.checks import valid_signature

            body = json.dumps(
                {
                    "event_id": f"e{i}",
                    "order_id": "00000000-0000-0000-0000-000000000000",
                    "amount": "1.00",
                    "status": "succeeded",
                }
            ).encode()
            return c.post(
                "/webhooks/payment",
                content=body,
                headers={"X-Signature": valid_signature(body), "Content-Type": "application/json"},
            ).status_code

        t0 = time.perf_counter()
        with ThreadPoolExecutor(20) as pool:
            codes = list(pool.map(call, range(40)))
        assert set(codes) == {404}
        return time.perf_counter() - t0


def test_p04_blocking_handler() -> None:
    clean, buggy = _webhook_burst_s([]), _webhook_burst_s(["P04"])
    assert buggy > 3 * clean, (clean, buggy)  # 40 x 50 ms serialised vs overlapped


def _error_ratio_under_concurrency(perf_bugs: list[str]) -> float:
    limits = httpx.Limits(max_connections=40)
    with (
        TargetServer(perf_bugs=perf_bugs) as t,
        httpx.Client(base_url=t.base_url, limits=limits, timeout=30) as c,
    ):
        t.seed(20_000)

        def call(_: int) -> int:
            return c.get("/orders", params={"page_size": 100}, headers=ADMIN).status_code

        with ThreadPoolExecutor(24) as pool:
            codes = list(pool.map(call, range(120)))
        return sum(c >= 500 for c in codes) / len(codes)


def test_p05_pool_exhaustion() -> None:
    clean, buggy = _error_ratio_under_concurrency([]), _error_ratio_under_concurrency(["P05"])
    assert clean == 0.0
    assert buggy > 0.05, buggy


def _rss_growth_mb(perf_bugs: list[str]) -> float:
    with TargetServer(perf_bugs=perf_bugs) as t:
        with httpx.Client(base_url=t.base_url, headers=ADMIN) as c:
            for _ in range(50):
                c.get("/health")
            before = float(t.metrics()["rss_mb"])  # type: ignore[arg-type]
            for _ in range(600):
                c.get("/health")
        return float(t.metrics()["rss_mb"]) - before  # type: ignore[arg-type]


def test_p06_memory_growth() -> None:
    clean, buggy = _rss_growth_mb([]), _rss_growth_mb(["P06"])
    assert buggy > 8, buggy  # 600 x 20 KB retained
    assert buggy > 4 * max(clean, 1.0)
