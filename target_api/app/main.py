"""Orders and Invoicing API: the system under test.

A deliberately small FastAPI service with SQLite storage, token auth for three roles (admin,
staff, customer) and seeded defects toggled by ``BUGS`` / ``PERF_BUGS`` (see ``bugs.yaml`` and
``perf_bugs.yaml``). The OpenAPI document is identical for every build; only behaviour changes.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from opentelemetry import trace

from target_api.app import bugs, routes_catalog, routes_orders, routes_payments, routes_testenv
from target_api.app.db import PoolExhausted, RequestStats, request_stats
from target_api.app.state import LOG_PATH, _leak_cache, _log, _route_stats, _stats_lock, tracer


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


for _module in (routes_catalog, routes_orders, routes_payments, routes_testenv):
    app.include_router(_module.router)
