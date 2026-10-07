"""Kill switch: a file or an env flag, checked between steps and before every LLM call.

What it stops: a runaway run (looping agent, unexpected spend, a load test hurting a target).
``agentqa killswitch`` creates the file; every running pipeline aborts cleanly at its next
check with a partial report.
"""

from __future__ import annotations

import os
from pathlib import Path

from agentqa import config
from agentqa.config import REPO_ROOT
from agentqa.llm.types import CallMetadata, KillSwitchEngaged
from agentqa.obs import metrics, tracing


def killswitch_path() -> Path:
    name = os.environ.get("AGENTQA_KILLSWITCH_FILE", config.guardrails().killswitch.file)
    path = Path(name)
    return path if path.is_absolute() else REPO_ROOT / path


def engaged() -> bool:
    env_flag = config.guardrails().killswitch.env
    return os.environ.get(env_flag, "0") == "1" or killswitch_path().exists()


def check(where: str) -> None:
    if engaged():
        metrics.inc("agentqa_guardrail_events_total", guardrail="killswitch", action="abort")
        tracing.event("guardrail.killswitch", where=where, action="abort")
        raise KillSwitchEngaged(f"kill switch engaged (checked at {where})")


def engage(reason: str = "manual") -> Path:
    path = killswitch_path()
    path.write_text(reason + "\n", encoding="utf-8")
    return path


def release() -> None:
    killswitch_path().unlink(missing_ok=True)


def llm_hook(meta: CallMetadata, est_tokens: int) -> None:
    check(f"llm_call:{meta.agent}")
