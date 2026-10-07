"""Named delegation strategies for the cost-versus-quality comparison (brief section 9).

S0 strong-only: the strongest tier for every LLM step, every cost mechanism off (quality
   ceiling and naive cost baseline).
S1 cheap-only: the cheapest tier for every LLM step, every cost mechanism off.
S2 static role routing: the profile's role -> tier mapping, every cost mechanism off.
S3 smart dispatcher: every mechanism in config/dispatch.yaml on.
Ablations: S3 with exactly one mechanism switched off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentqa import config


@dataclass(frozen=True)
class Strategy:
    name: str
    mechanisms: dict[str, bool]
    forced_tier: str | None = None
    description: str = ""
    overrides: dict[str, bool] = field(default_factory=dict)


def _all(value: bool) -> dict[str, bool]:
    return {k: value for k in config.dispatch().mechanisms}


def get(name: str) -> Strategy:
    if name == "S0":
        return Strategy("S0", _all(False), "T2", "strong-only, no cost mechanisms")
    if name == "S1":
        return Strategy("S1", _all(False), "T1", "cheap-only, no cost mechanisms")
    if name == "S2":
        return Strategy(
            "S2", _all(False), None, "static role routing (profile), no cost mechanisms"
        )
    if name == "S3":
        return Strategy(
            "S3", dict(config.dispatch().mechanisms), None, "smart dispatcher, all mechanisms on"
        )
    if name.startswith("S3-no-"):
        mech = name[len("S3-no-") :].replace("-", "_")
        base = dict(config.dispatch().mechanisms)
        if mech not in base:
            raise KeyError(f"unknown mechanism {mech}")
        base[mech] = False
        return Strategy(name, base, None, f"S3 without {mech}", {mech: False})
    raise KeyError(f"unknown strategy {name!r}")


ABLATIONS = [
    "deterministic_first",
    "cascade",
    "batching",
    "dedupe",
    "incremental",
    "cluster_first_triage",
    "caching",
    "context_budgeting",
    "learned_routing",
    "budget_allocation",
]
