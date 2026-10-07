"""F10 web UI: run history, run detail (stage timeline, findings with evidence, guardrail events,
cost, Phoenix link) and the eval scoreboard. Server-rendered Jinja; no JS framework.

    uv run agentqa serve   ->  http://127.0.0.1:8088
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from agentqa.config import REPO_ROOT, agentqa_home
from agentqa.store import Store

app = FastAPI(title="AgentQA")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
PHOENIX = "http://localhost:6006"


def _run_dir(run_id: str) -> Path:
    root = (agentqa_home() / "runs").resolve()
    path = (root / run_id).resolve()
    if path.parent != root or not path.exists():
        raise HTTPException(404, "run not found")
    return path


def timeline(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "spans.jsonl"
    if not path.exists():
        return []
    spans = [json.loads(line) for line in path.read_text().splitlines()]
    shown = [
        s for s in spans if s["attributes"].get("agentqa.span_kind") in ("run", "stage", "agent")
    ]
    if not shown:
        return []
    t0 = min(s["start_ns"] for s in shown)
    t1 = max(s["end_ns"] for s in shown)
    total = max(1, t1 - t0)
    rows = []
    for s in sorted(shown, key=lambda s: s["start_ns"])[:120]:
        rows.append(
            {
                "name": s["name"],
                "kind": s["attributes"].get("agentqa.span_kind"),
                "left": round((s["start_ns"] - t0) / total * 100, 2),
                "width": max(0.4, round((s["end_ns"] - s["start_ns"]) / total * 100, 2)),
                "ms": round((s["end_ns"] - s["start_ns"]) / 1e6, 1),
                "status": s["status"],
            }
        )
    return rows


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    runs = Store().runs(100)
    for r in runs:
        r["summary_obj"] = json.loads(r["summary"]) if r.get("summary") else {}
    return templates.TemplateResponse(request, "index.html", {"runs": runs})


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: str) -> HTMLResponse:
    d = _run_dir(run_id)
    report = json.loads((d / "report.json").read_text()) if (d / "report.json").exists() else None
    stats = json.loads((d / "stats.json").read_text()) if (d / "stats.json").exists() else {}
    delegations = [
        dict(r)
        for r in Store().query(
            "SELECT task_id, task_type, start_tier, reason, actual_tokens, verifier_outcome, escalation_path, final_result "
            "FROM delegations WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
    ]
    return templates.TemplateResponse(
        request,
        "run.html",
        {
            "run_id": run_id,
            "r": report,
            "stats": stats,
            "timeline": timeline(d),
            "delegations": delegations,
            "phoenix": f"{PHOENIX}/projects",
            "trace_id": (report or {}).get("trace_id"),
        },
    )


@app.get("/runs/{run_id}/report.html")
def run_report(run_id: str) -> FileResponse:
    return FileResponse(_run_dir(run_id) / "report.html")


@app.get("/runs/{run_id}/summary.html")
def run_summary(run_id: str) -> FileResponse:
    return FileResponse(_run_dir(run_id) / "executive_summary.html")


@app.get("/scoreboard", response_class=HTMLResponse)
def scoreboard(request: Request) -> HTMLResponse:
    rows = []
    for path in sorted((REPO_ROOT / "results").glob("*.json"), reverse=True):
        data = json.loads(path.read_text())
        for name, d in (data.get("strategies") or {}).items():
            a = d["aggregate"]
            rows.append(
                {
                    "file": path.name,
                    "commit": data.get("commit"),
                    "strategy": name,
                    "n": a["n"],
                    "recall": a["recall"]["mean"],
                    "precision": (a.get("precision") or {}).get("mean"),
                    "triage": (a.get("triage_accuracy") or {}).get("mean"),
                    "cost": a["cost_usd_list_equivalent"]["mean"],
                    "tokens": a["tokens_total"]["mean"],
                }
            )
    return templates.TemplateResponse(request, "scoreboard.html", {"rows": rows})


@app.get("/api/runs")
def api_runs() -> list[dict[str, Any]]:
    return Store().runs(100)
