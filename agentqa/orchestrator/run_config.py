"""Configuration and output of one AgentQA run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentqa.agents.reporter import RunReport
from agentqa.dispatch.cascade import (
    DelegationLedger,
)
from agentqa.ingest.vectorstore import VectorStore
from agentqa.llm.pricing import CostLedger
from agentqa.models import RunResult, SpecBundle, ValidatedTest
from agentqa.store import Store
from agentqa.target_config import TargetConfig


@dataclass
class RunConfig:
    spec: str | Path
    docs: str | Path | None
    target: TargetConfig
    base_url: str
    reference_url: str | None = None
    profile: str = "simulated"
    strategy: str = "S3"
    mechanism_overrides: dict[str, bool] = field(default_factory=dict)
    run_id: str | None = None
    out_root: Path | None = None
    budget: dict[str, float] = field(default_factory=dict)
    cache_mode: str | None = None
    use_lessons: bool = True
    server_log: Callable[[float], list[str]] | None = None
    store: Store | None = None
    vector_store: VectorStore | None = None
    concurrency: int | None = None
    judge: bool = True
    phoenix_base: str | None = "http://localhost:6006"


@dataclass
class RunOutput:
    run_id: str
    run_dir: Path
    report: RunReport
    paths: dict[str, Path]
    run_result: RunResult | None
    ledger: CostLedger
    delegation: DelegationLedger
    stats: dict[str, Any]
    tests: list[ValidatedTest] = field(default_factory=list)
    bundle: SpecBundle | None = None
