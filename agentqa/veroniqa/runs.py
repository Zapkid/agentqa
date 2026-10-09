"""Test runs started from a VeroniQA project, using AgentQA's own pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agentqa.config import REPO_ROOT
from agentqa.veroniqa.projects import Project

DEMO_TARGET = REPO_ROOT / "target_api/agentqa_target.yaml"
BUG_ID = re.compile(r"\bB(0?[1-9]|1[0-2])\b", re.IGNORECASE)


class NotRunnable(ValueError):
    pass


class RunSummary(BaseModel):
    run_id: str
    created_at: str = ""
    outcomes: dict[str, int] = {}
    findings: int = 0
    product_bugs: int = 0
    tokens: int = 0
    report_html: str = ""
    summary_md: str = ""


def demo_build(bugs: str) -> str:
    """Normalise a bug selection for the bundled API: "all", "clean" or "B01,B04"."""
    text = (bugs or "").strip().lower()
    if text in ("", "all", "every", "all bugs"):
        return "all"
    if text in ("clean", "none", "no bugs"):
        return "clean"
    ids = sorted({f"B{int(m.group(1)):02d}" for m in BUG_ID.finditer(text)})
    return ",".join(ids) if ids else "all"


def run_project_tests(project: Project, profile: str, bugs: str = "") -> Any:
    """Run the full AgentQA pipeline for a project. The project's documents are the requirement
    docs. Returns the pipeline's RunOutput."""
    from agentqa.cli import execute_run

    cfg = project.config
    docs = str(project.docs_dir) if any(project.docs_dir.glob("*.md")) else None
    project.runs_dir.mkdir(parents=True, exist_ok=True)
    if cfg.local_demo:
        return execute_run(
            spec=cfg.spec,
            docs=docs,
            profile=profile,
            strategy="S3",
            target_config=Path(cfg.target_config or DEMO_TARGET),
            base_url=None,
            reference_url=None,
            local_target=demo_build(bugs),
            local_reference=True,
            cache_mode=None,
            run_id=None,
            max_tokens=None,
            out_root=project.runs_dir,
        )
    missing = [k for k in ("spec", "base_url", "target_config") if not getattr(cfg, k)]
    if missing:
        raise NotRunnable(
            "This project cannot run tests yet: set "
            + ", ".join(missing)
            + " in the project settings (the target config says how to authenticate and whether "
            "the API is a sandbox that may be written to)."
        )
    return execute_run(
        spec=cfg.spec,
        docs=docs,
        profile=profile,
        strategy="S3",
        target_config=Path(str(cfg.target_config)),
        base_url=cfg.base_url,
        reference_url=None,
        local_target=None,
        local_reference=False,
        cache_mode=None,
        run_id=None,
        max_tokens=None,
        out_root=project.runs_dir,
    )


def project_runs(project: Project) -> list[RunSummary]:
    out: list[RunSummary] = []
    if not project.runs_dir.exists():
        return out
    for d in project.runs_dir.iterdir():
        report = d / "report.json"
        if not report.exists():
            continue
        r = json.loads(report.read_text(encoding="utf-8"))
        findings = r.get("findings", [])
        outcomes: dict[str, int] = {}
        for t in r.get("results", []):
            outcomes[t["outcome"]] = outcomes.get(t["outcome"], 0) + 1
        out.append(
            RunSummary(
                run_id=r["run_id"],
                created_at=r.get("created_at", ""),
                outcomes=outcomes,
                findings=len(findings),
                product_bugs=sum(f.get("classification") == "product_bug" for f in findings),
                tokens=int(r.get("cost", {}).get("total_tokens", 0)),
                report_html=str(d / "report.html"),
                summary_md=str(d / "executive_summary.md"),
            )
        )
    return sorted(out, key=lambda s: s.created_at, reverse=True)
