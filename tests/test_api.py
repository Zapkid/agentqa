from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from agentqa.api.app import app
from agentqa.config import agentqa_home
from agentqa.store import Store


def test_ui_pages(tmp_path: Path) -> None:
    run_dir = agentqa_home() / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "simulated": True,
                "aborted": None,
                "api_title": "API",
                "api_version": "1",
                "profile": "simulated",
                "strategy": "S3",
                "target": "http://t",
                "trace_id": "abc",
                "quarantined_chunks": [],
                "guardrail_events": [],
                "cost": {
                    "total_tokens": 1,
                    "cached_input_tokens": 0,
                    "calls": 1,
                    "cost_usd_actual": 0,
                    "cost_usd_list_equivalent": 0.1,
                },
                "delegation": {"escalations": 0, "tokens_saved_estimate": 5},
                "findings": [
                    {
                        "id": "F-1",
                        "severity": "high",
                        "classification": "product_bug",
                        "title": "IDOR <script>",
                        "endpoint": "GET /x",
                        "confidence": 0.9,
                        "triaged_by": "T1",
                        "root_cause_hypothesis": "h",
                        "repro_curl": "curl x",
                        "evidence": [{"ref": "e", "kind": "request", "excerpt": "x"}],
                    }
                ],
            }
        )
    )
    (run_dir / "spans.jsonl").write_text(
        json.dumps(
            {
                "name": "run r1",
                "start_ns": 0,
                "end_ns": 10,
                "status": "OK",
                "attributes": {"agentqa.span_kind": "run"},
            }
        )
        + "\n"
    )
    Store().save_run(
        "r1", status="done", profile="simulated", strategy="S3", summary={"findings": 1}
    )
    c = TestClient(app)
    assert "r1" in c.get("/").text
    page = c.get("/runs/r1").text
    assert "IDOR &lt;script&gt;" in page and "Simulated models" in page and "run r1" in page
    assert c.get("/runs/../../etc").status_code == 404
    assert c.get("/scoreboard").status_code == 200
