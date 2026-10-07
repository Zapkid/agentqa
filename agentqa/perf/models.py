"""Performance artifacts: the LLM writes a WorkloadSpec; deterministic code does the rest."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TestType = Literal["smoke", "load", "stress", "spike", "soak"]
BottleneckClass = Literal[
    "n_plus_one",
    "missing_index",
    "unbounded_payload",
    "blocking_handler",
    "pool_exhaustion",
    "memory_leak",
    "none",
]


class Scenario(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9_]{2,40}$")
    method: Literal["GET", "POST", "PATCH", "PUT", "DELETE"]
    path: str  # spec template, e.g. /orders/{order_id}
    params: dict[str, str | int] = Field(default_factory=dict)
    body: Literal["none", "example", "order", "webhook"] = "none"
    weight: int = Field(ge=1, le=100)
    expected_status: list[int] = Field(min_length=1)


class SLOs(BaseModel):
    p95_ms: float | None = None
    p99_ms: float | None = None
    error_rate: float | None = None
    min_rps: float | None = None
    scope: str = Field(
        default="", description="what the SLO applies to, e.g. a scenario name or 'all'"
    )


class WorkloadSpec(BaseModel):
    scenarios: list[Scenario] = Field(min_length=1, max_length=12)
    think_time_s: float = Field(ge=0, le=5)
    data_setup_orders: int = Field(ge=0, le=500_000)
    ramp_profile: str = Field(max_length=200)
    slos: list[SLOs] = Field(default_factory=list, max_length=6)
    slo_source: list[str] = Field(
        default_factory=list
    )  # chunk ids the SLO numbers were quoted from
    peak_users: int = Field(default=20, ge=1, le=500)


class StepStats(BaseModel):
    users: int
    scenario: str
    count: int
    rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    error_rate: float
    errors_by_status: dict[str, int] = Field(default_factory=dict)
    avg_bytes: float


class IterationResult(BaseModel):
    iteration: int
    steps: list[StepStats]
    target_before: dict[str, Any]
    target_after: dict[str, Any]
    wall_s: float
    aborted: str | None = None


class PerfRun(BaseModel):
    label: str  # build label, e.g. clean or P01
    test_type: str
    workload: WorkloadSpec
    iterations: list[IterationResult]
    host: dict[str, object] = Field(default_factory=dict)


class PerfVerdict(BaseModel):
    """What the perf-triage model returns for one evidence bundle."""

    bottleneck: BottleneckClass
    confidence: float = Field(ge=0, le=1)
    evidence_keys: list[str] = Field(default_factory=list, max_length=10)
    explanation: str = Field(max_length=800)
    fix: str = Field(max_length=600)
