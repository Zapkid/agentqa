"""Artifacts exchanged between agents (strict pydantic contracts).

Agents never call each other; the supervisor passes these objects between them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

CATEGORIES = [
    "validation",
    "boundary",
    "auth/permission",
    "state-transition",
    "idempotency",
    "error-handling",
    "data-leak",
    "contract-conformance",
    "money/precision",
    "webhooks/signatures",
    "time/timezone",
    "business-logic",
]
Category = Literal[
    "validation",
    "boundary",
    "auth/permission",
    "state-transition",
    "idempotency",
    "error-handling",
    "data-leak",
    "contract-conformance",
    "money/precision",
    "webhooks/signatures",
    "time/timezone",
    "business-logic",
]

# ------------------------------------------------------------------ ingestion


class Param(BaseModel):
    name: str
    location: Literal["path", "query", "header", "cookie"]
    required: bool = False
    schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")
    description: str = ""

    model_config = {"populate_by_name": True}


class Endpoint(BaseModel):
    id: str  # "GET /orders/{order_id}"
    method: str
    path: str
    operation_id: str = ""
    summary: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    params: list[Param] = Field(default_factory=list)
    request_schema: dict[str, Any] | None = None
    responses: dict[str, dict[str, Any] | None] = Field(default_factory=dict)
    requires_auth: bool = False
    auth_schemes: list[str] = Field(default_factory=list)

    @property
    def status_codes(self) -> set[str]:
        return set(self.responses)

    def body_fields(self) -> dict[str, Any]:
        props: dict[str, Any] = (self.request_schema or {}).get("properties", {})
        return props


class Chunk(BaseModel):
    id: str
    kind: Literal["endpoint", "doc"]
    text: str
    endpoint_id: str | None = None
    doc: str | None = None
    section: str | None = None
    sha: str = ""
    quarantined: bool = False
    quarantine_reason: str | None = None


class SpecBundle(BaseModel):
    spec_id: str
    title: str
    version: str
    endpoints: list[Endpoint]
    chunks: list[Chunk]
    endpoint_hashes: dict[str, str] = Field(default_factory=dict)  # for diff-aware runs

    def endpoint(self, endpoint_id: str) -> Endpoint:
        for ep in self.endpoints:
            if ep.id == endpoint_id:
                return ep
        raise KeyError(endpoint_id)


# ------------------------------------------------------------------ planning


class TestIntent(BaseModel):
    __test__ = False  # not a pytest class

    id: str
    endpoint: str
    category: Category
    title: str
    risk: int = Field(ge=1, le=5)
    rationale: str
    preconditions: list[str] = Field(default_factory=list)
    expected_behavior: str
    source_refs: list[str] = Field(min_length=1)
    origin: Literal["t0", "llm"] = "llm"

    @field_validator("id")
    @classmethod
    def _id_safe(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("intent id must be alphanumeric with - or _")
        return v


class TestPlan(BaseModel):
    __test__ = False

    intents: list[TestIntent]


class PlannedIntent(BaseModel):
    """What the planner model returns per intent (ids are assigned by code)."""

    category: Category
    title: str = Field(max_length=160)
    risk: int = Field(ge=1, le=5)
    rationale: str = Field(max_length=600)
    preconditions: list[str] = Field(default_factory=list, max_length=6)
    expected_behavior: str = Field(max_length=600)
    source_refs: list[str] = Field(min_length=1, max_length=6)


class PlannerOutput(BaseModel):
    intents: list[PlannedIntent] = Field(max_length=12)


# ------------------------------------------------------------------ generation


class RequestDecl(BaseModel):
    method: str
    path: str  # template, e.g. /orders/{order_id}
    fields: list[str] = Field(default_factory=list)  # body/query field names touched
    expected_status: list[int] = Field(default_factory=list)


class GeneratedTest(BaseModel):
    __test__ = False

    intent_id: str
    test_name: str = Field(pattern=r"^test_[a-z0-9_]{3,80}$")
    code: str = Field(max_length=6000)
    requests_made: list[RequestDecl] = Field(min_length=1, max_length=12)
    confidence: float = Field(default=0.8, ge=0, le=1)


class GeneratorOutput(BaseModel):
    test_name: str = Field(pattern=r"^test_[a-z0-9_]{3,80}$")
    code: str = Field(max_length=6000)
    requests_made: list[RequestDecl] = Field(min_length=1, max_length=12)
    confidence: float = Field(ge=0, le=1)


class GuardViolation(BaseModel):
    kind: str  # unknown_path | method_not_allowed | unknown_field | undocumented_status | static | ...
    detail: str


class ValidatedTest(BaseModel):
    """A generated test after static checks and the grounding guard."""

    intent: TestIntent
    test: GeneratedTest
    tier: str  # T0 | T1 | T2
    violations_pre_repair: list[GuardViolation] = Field(default_factory=list)
    repaired: bool = False
    quarantined: bool = False
    escalations: list[str] = Field(default_factory=list)
    reused: bool = False  # from a previous run (incremental)


# ------------------------------------------------------------------ execution and triage


class HttpExchange(BaseModel):
    method: str
    url: str
    path: str
    status: int | None
    request_headers: dict[str, str] = Field(default_factory=dict)
    request_body: str | None = None
    response_body: str | None = None
    elapsed_ms: float = 0.0
    blocked: str | None = None  # guardrail reason if the sandbox blocked it


class TestResult(BaseModel):
    __test__ = False

    test_name: str
    intent_id: str
    outcome: Literal["passed", "failed", "error", "skipped", "blocked"]
    message: str = ""
    duration_s: float = 0.0
    exchanges: list[HttpExchange] = Field(default_factory=list)
    reruns: list[str] = Field(default_factory=list)  # outcomes of flaky reruns

    @property
    def flaky(self) -> bool:
        """Pass/fail changed across reruns. A skipped rerun is inconclusive, not a flip."""
        decisive = {o for o in [self.outcome, *self.reruns] if o in ("passed", "failed", "error")}
        return bool(self.reruns) and len(decisive) > 1


class RunResult(BaseModel):
    target: str
    base_url: str
    results: list[TestResult]
    started_ts: float
    ended_ts: float
    server_log: list[str] = Field(default_factory=list)
    guardrail_events: list[dict[str, Any]] = Field(default_factory=list)


FindingClass = Literal["product_bug", "test_bug", "flaky", "env_issue", "needs_review"]
Severity = Literal["critical", "high", "medium", "low"]


class Evidence(BaseModel):
    kind: Literal["request", "response", "spec_clause", "log_line", "doc_clause", "rerun"]
    ref: str  # artifact path or id
    excerpt: str = Field(max_length=1500)


class TriageVerdict(BaseModel):
    """What the triage model returns for one failure cluster."""

    classification: Literal["product_bug", "test_bug", "flaky", "env_issue"]
    severity: Severity
    root_cause_hypothesis: str = Field(max_length=800)
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)
    confidence: float = Field(ge=0, le=1)
    title: str = Field(max_length=160)


class Finding(BaseModel):
    id: str
    title: str
    classification: FindingClass
    severity: Severity
    endpoint: str
    category: str
    root_cause_hypothesis: str
    evidence: list[Evidence]
    repro_curl: str
    test_names: list[str]
    intent_ids: list[str]
    confidence: float
    triaged_by: str  # tier/model or "rule"
    endpoints: list[str] = Field(default_factory=list)  # all endpoints sharing this root cause
    judge_checked: bool = False
