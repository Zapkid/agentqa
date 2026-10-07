"""Typed loaders for the YAML files in ``config/``.

Everything tunable lives in YAML; this module only parses and validates it.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("AGENTQA_CONFIG_DIR", REPO_ROOT / "config"))


def agentqa_home() -> Path:
    home = Path(os.environ.get("AGENTQA_HOME", REPO_ROOT / ".agentqa"))
    home.mkdir(parents=True, exist_ok=True)
    return home


def _load(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{name} must contain a mapping")
    return data


# ---------------------------------------------------------------- providers


class ProviderConfig(BaseModel):
    api_key_env: str | None = None
    base_url: str | None = None
    free_tier: bool = False
    rpm: int
    tpm: int
    rpd: int
    max_retries: int = 3
    native_structured_output: bool = True
    native_prompt_cache: str = "none"


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = 3
    reset_after_s: float = 60.0


class BackoffConfig(BaseModel):
    base_s: float = 1.0
    max_s: float = 30.0
    jitter: float = 0.25


class StructuredOutputConfig(BaseModel):
    max_repair_retries: int = 2


class ProvidersFile(BaseModel):
    providers: dict[str, ProviderConfig]
    circuit_breaker: CircuitBreakerConfig = CircuitBreakerConfig()
    backoff: BackoffConfig = BackoffConfig()
    structured_output: StructuredOutputConfig = StructuredOutputConfig()


# ---------------------------------------------------------------- pricing


class ModelPrice(BaseModel):
    input: float
    output: float
    cached_input: float
    cache_write: float
    as_of: str
    verify_before_use: bool = False
    note: str | None = None


class PricingFile(BaseModel):
    as_of: str
    models: dict[str, ModelPrice]


# ---------------------------------------------------------------- models / profiles


class ModelRef(BaseModel):
    provider: str
    model: str


class TierConfig(ModelRef):
    fallbacks: list[ModelRef] = Field(default_factory=list)

    def chain(self) -> list[ModelRef]:
        return [ModelRef(provider=self.provider, model=self.model), *self.fallbacks]


class Profile(BaseModel):
    tiers: dict[str, TierConfig]
    roles: dict[str, str]
    judge: ModelRef

    def tier_for_role(self, role: str) -> str:
        return self.roles.get(role, "T1")


class ModelsFile(BaseModel):
    as_of: str
    families: dict[str, str]
    roles: list[str]
    profiles: dict[str, Profile]

    @model_validator(mode="after")
    def judge_family_differs(self) -> ModelsFile:
        for name, profile in self.profiles.items():
            judge_family = self.families.get(profile.judge.model)
            if judge_family is None:
                raise ValueError(f"profile {name}: judge model {profile.judge.model} has no family")
            for tier_name, tier in profile.tiers.items():
                if tier_name not in {"T1", "T2"}:
                    raise ValueError(f"profile {name}: unknown tier {tier_name}")
                for ref in tier.chain():
                    if self.families.get(ref.model) == judge_family:
                        raise ValueError(
                            f"profile {name}: judge family {judge_family} equals generator "
                            f"family of {ref.model} (self-preference bias)"
                        )
        return self


# ---------------------------------------------------------------- guardrails


class BudgetConfig(BaseModel):
    max_tokens: int
    max_usd_list_equivalent: float
    max_wall_clock_s: float
    max_agent_steps: int


class SandboxConfig(BaseModel):
    timeout_s: int = 120
    memory_mb: int = 1024
    cpu_s: int = 120
    allow_mutations: bool = False
    read_only_methods: list[str] = Field(default_factory=lambda: ["GET", "HEAD", "OPTIONS"])
    allowed_imports: list[str]
    banned_calls: list[str]


class TargetConfig(BaseModel):
    base_url_pattern: str
    sandbox: bool = False


class InjectionConfig(BaseModel):
    quarantine: bool = True
    heuristic_threshold: int = 1
    classifier: bool = True


class LoadGuardConfig(BaseModel):
    max_users: int
    max_rps: int
    max_duration_s: int
    abort_error_rate: float
    abort_host_cpu_pct: float
    abort_host_mem_pct: float


class KillswitchConfig(BaseModel):
    file: str = "KILLSWITCH"
    env: str = "AGENTQA_KILL"


class GuardrailsFile(BaseModel):
    budget: BudgetConfig
    sandbox: SandboxConfig
    targets: dict[str, TargetConfig]
    injection: InjectionConfig
    load: LoadGuardConfig
    killswitch: KillswitchConfig


# ---------------------------------------------------------------- dispatch


class DispatchParams(BaseModel):
    batch_size: int = 4
    dedupe_similarity: float = 0.85
    retrieval_top_k: int = 5
    retrieval_token_cap: int = 1200
    max_intents_per_endpoint: int = 6
    strong_tier_floor: float = 0.05
    difficulty_t2_threshold: float = 0.6
    triage_escalate_confidence: float = 0.6
    concurrency: int = 4


class DispatchFile(BaseModel):
    mechanisms: dict[str, bool]
    params: DispatchParams
    llm_categories: list[str]
    t0_categories: list[str]

    def on(self, mechanism: str) -> bool:
        if mechanism not in self.mechanisms:
            raise KeyError(f"unknown dispatch mechanism {mechanism!r}")
        return self.mechanisms[mechanism]

    def with_overrides(self, **flags: bool) -> DispatchFile:
        merged = {**self.mechanisms, **flags}
        return self.model_copy(update={"mechanisms": merged})


# ---------------------------------------------------------------- perf


class PerfDefaults(BaseModel):
    slos: dict[str, float]
    shapes: dict[str, list[list[float]]]
    warmup_s: float = 1.5
    iterations: int = 3
    data_setup_orders_cap: int = 200_000


# ---------------------------------------------------------------- accessors


@lru_cache
def providers() -> ProvidersFile:
    return ProvidersFile.model_validate(_load("providers.yaml"))


@lru_cache
def pricing() -> PricingFile:
    return PricingFile.model_validate(_load("pricing.yaml"))


@lru_cache
def models() -> ModelsFile:
    return ModelsFile.model_validate(_load("models.yaml"))


@lru_cache
def guardrails() -> GuardrailsFile:
    return GuardrailsFile.model_validate(_load("guardrails.yaml"))


@lru_cache
def dispatch() -> DispatchFile:
    return DispatchFile.model_validate(_load("dispatch.yaml"))


@lru_cache
def perf_defaults() -> PerfDefaults:
    return PerfDefaults.model_validate(_load("perf_defaults.yaml"))


def profile(name: str | None = None) -> Profile:
    name = name or os.environ.get("AGENTQA_PROFILE", "simulated")
    try:
        return models().profiles[name]
    except KeyError as exc:
        raise KeyError(f"unknown profile {name!r}; known: {sorted(models().profiles)}") from exc
