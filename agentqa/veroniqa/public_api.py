"""VeroniQA's public API: a small, keyless, read-mostly sandbox mounted at /api.

    GET  /api/v1/status                 service status and version
    GET  /api/v1/results                the measured results shown on the home page
    POST /api/v1/ask                    a cited answer from the demo API's requirement documents
    GET  /api/v1/demo/openapi.json      the OpenAPI spec of the bundled demo (Orders) API

Conventions agents can rely on:

- The OpenAPI 3.1 description is served at /openapi.json, /openapi.yaml, /api/openapi.json and
  /api/openapi.yaml; /.well-known/api-catalog (RFC 9727) points to it.
- Every error under /api, including unknown paths, wrong methods and invalid input, is an RFC 9457
  problem document (`application/problem+json`) with a stable `code` and a `hint`.
- Every response carries rate-limit headers (draft-ietf-httpapi-ratelimit-headers: `RateLimit-Policy`
  and `RateLimit`), and a 429 adds `Retry-After`.

Models are simulated (no API keys on the server), so answers are extracts from the demo
documents, never a model's prose. No key is needed and nothing is stored.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from functools import cache
from typing import Any

import yaml
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from agentqa.config import REPO_ROOT, agentqa_home
from agentqa.veroniqa import site

API_VERSION = "1.0.0"
PROBLEM = "application/problem+json"
DEMO_SPEC = REPO_ROOT / "target_api" / "openapi.json"


# ------------------------------------------------------------------ models


class Problem(BaseModel):
    """An RFC 9457 problem document, with a stable machine-readable code and a hint."""

    type: str = Field(description="URI identifying the problem type; documented on /developers")
    title: str = Field(description="Short, human-readable summary of the problem type")
    status: int = Field(description="The HTTP status code")
    detail: str = Field(description="What went wrong in this request")
    instance: str = Field(description="The request path")
    code: str = Field(description="Stable error code, e.g. not_found or rate_limited")
    hint: str = Field(description="How to fix the request")
    errors: list[dict[str, Any]] | None = Field(
        default=None, description="Field-level validation errors (code invalid_request only)"
    )


class Status(BaseModel):
    status: str = Field(examples=["ok"])
    service: str = Field(examples=["VeroniQA public API"])
    version: str = Field(examples=[API_VERSION])
    models: str = Field(description="simulated: answers are extracts, not a model's prose")
    authentication: str = Field(examples=["none"])
    docs: str
    openapi: str


class Tile(BaseModel):
    value: str = Field(examples=["94%"])
    caption: str = Field(examples=["of planted bugs found"])


class Results(BaseModel):
    benchmark: str = Field(description="Where the numbers come from")
    recall_pct: int = Field(description="Share of planted bugs found with cost-aware routing")
    cost_pct: int = Field(description="Cost relative to using the strongest model for everything")
    perf_caught: int = Field(description="Performance defects caught")
    perf_total: int = Field(description="Performance defects planted")
    planted_bugs: int = Field(description="Functional bugs planted in the demo API")
    cost_levers: int = Field(description="Cost mechanisms measured one at a time (ablations)")
    tiles: list[Tile]


class AskIn(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=500,
        description="A question about the demo Orders API",
        examples=["What is the p95 latency target for the order list?"],
    )


class Citation(BaseModel):
    n: int = Field(description="Passage number as cited in the answer, e.g. [2]")
    source: str = Field(description="Document the passage comes from")
    section: str = Field(description="Heading the passage sits under")
    text: str = Field(description="The passage")


class AskOut(BaseModel):
    answer: str
    found: bool = Field(description="False when the documents do not answer the question")
    grounded: bool = Field(description="True when every cited passage was actually retrieved")
    citations: list[Citation]
    note: str = Field(default="", description="Caveats, e.g. dropped citations")
    knowledge: str = Field(description="The knowledge the answer is drawn from")


# ------------------------------------------------------------------ rate limits


class RateLimiter:
    """Fixed-window limits per client and bucket, reported with IETF RateLimit headers."""

    def __init__(self, limits: dict[str, tuple[int, int]]) -> None:
        self.limits = limits  # bucket -> (requests, window seconds)
        self.hits: dict[tuple[str, str], deque[float]] = {}
        self.lock = threading.Lock()

    def check(
        self, bucket: str, client: str, now: float | None = None
    ) -> tuple[bool, dict[str, str]]:
        quota, window = self.limits[bucket]
        now = time.monotonic() if now is None else now
        with self.lock:
            hits = self.hits.setdefault((bucket, client), deque())
            while hits and now - hits[0] >= window:
                hits.popleft()
            allowed = len(hits) < quota
            if allowed:
                hits.append(now)
            remaining = quota - len(hits)
            reset = max(1, math.ceil(window - (now - hits[0]))) if hits else window
            if len(self.hits) > 10_000:  # drop idle clients
                for key in [k for k, v in self.hits.items() if not v]:
                    del self.hits[key]
        headers = {
            "RateLimit-Policy": f'"{bucket}";q={quota};w={window}',
            "RateLimit": f'"{bucket}";r={remaining};t={reset}',
        }
        if not allowed:
            headers["Retry-After"] = str(reset)
        return allowed, headers


LIMITS = RateLimiter({"default": (60, 60), "ask": (10, 60)})


def client_id(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ------------------------------------------------------------------ problems


ERRORS: dict[str, tuple[int, str, str]] = {
    # code: (status, title, hint)
    "not_found": (
        404,
        "Not found",
        "Check the path against the OpenAPI description at /openapi.json.",
    ),
    "method_not_allowed": (
        405,
        "Method not allowed",
        "Use the method listed for this path in /openapi.json (see the Allow header).",
    ),
    "invalid_request": (
        422,
        "Invalid request",
        "Fix the fields listed in `errors`; the request schema is in /openapi.json.",
    ),
    "rate_limited": (
        429,
        "Too many requests",
        "Wait for the number of seconds in Retry-After, then retry.",
    ),
    "internal_error": (
        500,
        "Internal error",
        "Retry later; if it persists, open an issue at " + site.CONTACT_URL + ".",
    ),
}


def problem(
    request: Request,
    code: str,
    detail: str,
    *,
    headers: dict[str, str] | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    status, title, hint = ERRORS[code]
    body = Problem(
        type=f"{site.site_url()}/developers#error-{code}",
        title=title,
        status=status,
        detail=detail,
        instance=request.url.path,
        code=code,
        hint=hint,
        errors=errors,
    )
    return JSONResponse(
        body.model_dump(exclude_none=True),
        status_code=status,
        media_type=PROBLEM,
        headers=headers,
    )


def _problem_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    out: dict[int | str, dict[str, Any]] = {}
    for code, (status, title, _hint) in ERRORS.items():
        if status in statuses:
            out[status] = {
                "description": f"{title} (`{code}`)",
                "content": {PROBLEM: {"schema": {"$ref": "#/components/schemas/Problem"}}},
            }
    return out


# ------------------------------------------------------------------ the demo knowledge


_ask_lock = threading.Lock()


@cache
def _demo_agent() -> Any:
    """One shared, read-only copy of the demo project's knowledge base, built on first use."""
    from agentqa.veroniqa import VeroniQA, create_demo_project, list_projects

    root = agentqa_home() / "public-api" / "projects"
    projects = list_projects(root) or [create_demo_project(root)]
    return VeroniQA(projects[0], profile="simulated")


# ------------------------------------------------------------------ the app


api = FastAPI(
    title="VeroniQA public API",
    version=API_VERSION,
    summary="Ask VeroniQA about the demo API, read the measured results, and fetch the demo spec.",
    description=(
        "A keyless, rate-limited sandbox for the VeroniQA public demo. Answers come from the "
        "demo Orders API's requirement documents and cite their sources; models are simulated, "
        "so answers are extracts, not a model's prose. Errors are RFC 9457 problem documents "
        "(`application/problem+json`) with a stable `code` and a `hint`. Rate limits are "
        "reported with `RateLimit-Policy` and `RateLimit` headers. Guide: "
        f"{site.site_url()}/developers"
    ),
    contact={"name": site.AUTHOR, "url": site.CONTACT_URL},
    license_info={"name": "See repository", "url": site.REPO_URL},
    servers=[{"url": f"{site.site_url()}/api", "description": "Public demo (sandbox)"}],
    openapi_tags=[
        {"name": "service", "description": "Status and published results"},
        {"name": "knowledge", "description": "Cited answers from the demo documents"},
        {"name": "demo", "description": "The bundled demo API that VeroniQA tests"},
    ],
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    expose_headers=["RateLimit", "RateLimit-Policy", "Retry-After"],
)


@api.middleware("http")
async def rate_limit(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if request.method == "OPTIONS":
        return await call_next(request)
    bucket = "ask" if request.url.path.endswith("/v1/ask") else "default"
    allowed, headers = LIMITS.check(bucket, client_id(request))
    if not allowed:
        return problem(
            request,
            "rate_limited",
            f"More than the {bucket} limit for this client.",
            headers=headers,
        )
    response = await call_next(request)
    response.headers.update(headers)
    response.headers.setdefault("Cache-Control", "no-store")
    return response


@api.exception_handler(StarletteHTTPException)
async def http_problem(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    if exc.status_code == 405:
        return problem(
            request,
            "method_not_allowed",
            f"{request.method} is not supported on {request.url.path}.",
            headers=dict(exc.headers or {}),
        )
    if exc.status_code == 404:
        return problem(request, "not_found", f"There is no endpoint at {request.url.path}.")
    return problem(request, "internal_error", str(exc.detail))


@api.exception_handler(RequestValidationError)
async def validation_problem(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {
            "field": ".".join(str(p) for p in err.get("loc", ()) if p != "body"),
            "message": err.get("msg", ""),
        }
        for err in exc.errors()
    ]
    return problem(request, "invalid_request", "The request body is not valid.", errors=errors)


@api.exception_handler(Exception)
async def internal_problem(request: Request, exc: Exception) -> JSONResponse:
    return problem(request, "internal_error", "The request could not be completed.")


@api.get(
    "/",
    tags=["service"],
    operation_id="getIndex",
    summary="API index",
    description="Links to the API's description, guide and endpoints.",
    responses=_problem_responses(429, 500),
)
def get_index() -> dict[str, Any]:
    url = site.site_url()
    return {
        "service": "VeroniQA public API",
        "version": API_VERSION,
        "authentication": "none",
        "openapi": f"{url}/openapi.json",
        "docs": f"{url}/developers",
        "endpoints": {
            "status": f"{url}/api/v1/status",
            "results": f"{url}/api/v1/results",
            "ask": f"{url}/api/v1/ask",
            "demo_openapi": f"{url}/api/v1/demo/openapi.json",
        },
    }


@api.get(
    "/v1/status",
    response_model=Status,
    tags=["service"],
    operation_id="getStatus",
    summary="Service status",
    responses=_problem_responses(429, 500),
)
def get_status() -> Status:
    url = site.site_url()
    return Status(
        status="ok",
        service="VeroniQA public API",
        version=API_VERSION,
        models="simulated",
        authentication="none",
        docs=f"{url}/developers",
        openapi=f"{url}/openapi.json",
    )


@api.get(
    "/v1/results",
    response_model=Results,
    tags=["service"],
    operation_id="getResults",
    summary="Measured results",
    description="The numbers on the home page, from the latest benchmark results in the repository.",
    responses=_problem_responses(429, 500),
)
def get_results() -> Results:
    return Results(
        benchmark="simulated benchmark in the AgentQA repository (results/*.json)",
        **site.STATS,
        tiles=[Tile(value=v, caption=c) for _, v, c in site.stat_tiles()],
    )


@api.post(
    "/v1/ask",
    response_model=AskOut,
    tags=["knowledge"],
    operation_id="ask",
    summary="Ask about the demo API",
    description=(
        "Answers a question from the demo Orders API's requirement documents, with numbered "
        "citations. Returns found=false when the documents do not cover the question."
    ),
    responses=_problem_responses(422, 429, 500),
)
def ask(body: AskIn) -> AskOut:
    with _ask_lock:
        ans = _demo_agent().ask(body.question)
    return AskOut(
        answer=ans.text,
        found=bool(ans.citations),
        grounded=ans.grounded,
        citations=[
            Citation(n=p.n, source=p.source, section=p.section, text=p.text[:1200])
            for p in ans.citations
        ],
        note=ans.note,
        knowledge="Requirement documents of the bundled demo Orders API",
    )


@api.get(
    "/v1/demo/openapi.json",
    tags=["demo"],
    operation_id="getDemoSpec",
    summary="OpenAPI spec of the demo API",
    description="The OpenAPI description of the bundled Orders API that the demo tests run against.",
    responses={
        200: {"description": "An OpenAPI document", "content": {"application/json": {}}},
        **_problem_responses(429, 500),
    },
)
def get_demo_spec() -> Response:
    return Response(DEMO_SPEC.read_text(encoding="utf-8"), media_type="application/json")


@cache
def openapi_document() -> dict[str, Any]:
    doc = api.openapi()
    schemas = doc.setdefault("components", {}).setdefault("schemas", {})
    schemas["Problem"] = Problem.model_json_schema(ref_template="#/components/schemas/{model}")
    # FastAPI documents 422s as HTTPValidationError; this API answers them as problems.
    for path in doc.get("paths", {}).values():
        for op in path.values():
            if "422" in op.get("responses", {}):
                op["responses"]["422"] = {
                    "description": "Invalid request (`invalid_request`)",
                    "content": {PROBLEM: {"schema": {"$ref": "#/components/schemas/Problem"}}},
                }
    for unused in ("HTTPValidationError", "ValidationError"):
        schemas.pop(unused, None)
    return doc


def openapi_json() -> str:
    return json.dumps(openapi_document(), indent=2)


def openapi_yaml() -> str:
    return yaml.safe_dump(openapi_document(), sort_keys=False, allow_unicode=True)


def api_catalog() -> dict[str, Any]:
    """RFC 9727 API catalog, as an RFC 9264 linkset."""
    url = site.site_url()
    return {
        "linkset": [
            {
                "anchor": f"{url}/api",
                "service-desc": [
                    {"href": f"{url}/openapi.json", "type": "application/vnd.oai.openapi+json"},
                    {"href": f"{url}/openapi.yaml", "type": "application/vnd.oai.openapi"},
                ],
                "service-doc": [{"href": f"{url}/developers", "type": "text/html"}],
                "status": [{"href": f"{url}/api/v1/status", "type": "application/json"}],
            }
        ]
    }
