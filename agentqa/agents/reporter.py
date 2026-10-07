"""F7 Judge and reporter.

- Technical report (Markdown + self-contained HTML): outcomes, findings with evidence and curl
  repros, coverage by risk and category (including uncovered intents with the reason),
  guardrail events, cost ledger, delegation summary, judge scores.
- Customer executive summary in plain language. The manual-effort figure is an *estimate*,
  shown with its formula and assumptions, never presented as a fact.
- LLM judge (a different model family from the generator) scores the report on a rubric and
  lists findings whose claims it cannot match to evidence; those are downgraded to needs_review.
  Cheap verifiers first: only ambiguous findings (low confidence or high/critical severity) are
  sent to the judge.
"""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from jinja2 import Environment, select_autoescape
from pydantic import BaseModel, Field

from agentqa.guards.redaction import redact
from agentqa.llm.normalize import normalize
from agentqa.llm.prompts import load_prompt
from agentqa.llm.router import LLMClient
from agentqa.llm.types import LLMError
from agentqa.models import Finding, TestIntent, TestResult
from agentqa.obs import metrics, tracing

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
MINUTES_PER_TEST = 20  # assumption: design + write + run one API test case by hand
MINUTES_PER_FINDING = 30  # assumption: investigate, reproduce and write up one finding


class JudgeScores(BaseModel):
    accuracy: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    unsupported_finding_ids: list[str] = Field(default_factory=list)
    comments: str = Field(default="", max_length=1000)


class Uncovered(BaseModel):
    intent_id: str
    endpoint: str
    category: str
    risk: int
    reason: str


class RunReport(BaseModel):
    run_id: str
    created_at: str
    profile: str
    strategy: str
    simulated: bool
    api_title: str
    api_version: str
    target: str
    trace_id: str | None
    phoenix_url: str | None
    endpoints: int
    intents: list[TestIntent]
    results: list[TestResult]
    tiers: dict[str, str]  # intent_id -> tier
    findings: list[Finding]
    uncovered: list[Uncovered]
    guardrail_events: list[dict[str, Any]]
    quarantined_chunks: list[str]
    hallucination: dict[str, float]
    cost: dict[str, Any]
    delegation: dict[str, Any]
    judge: JudgeScores | None = None
    aborted: str | None = None
    incremental: dict[str, Any] = Field(default_factory=dict)
    lessons_used: int = 0

    # ---------------------------------------------------------------- derived views

    def outcome_counts(self) -> dict[str, int]:
        c = Counter(r.outcome for r in self.results)
        return {k: c.get(k, 0) for k in ("passed", "failed", "error", "skipped", "blocked")}

    def coverage(self) -> list[dict[str, Any]]:
        by_result = {r.intent_id: r for r in self.results}
        uncovered = {u.intent_id for u in self.uncovered}
        rows: dict[str, dict[str, Any]] = {}
        for i in self.intents:
            row = rows.setdefault(
                i.category,
                {
                    "category": i.category,
                    "planned": 0,
                    "tested": 0,
                    "passed": 0,
                    "failing": 0,
                    "uncovered": 0,
                    "max_risk": 0,
                },
            )
            row["planned"] += 1
            row["max_risk"] = max(row["max_risk"], i.risk)
            if i.id in uncovered:
                row["uncovered"] += 1
            elif i.id in by_result:
                row["tested"] += 1
                if by_result[i.id].outcome == "passed":
                    row["passed"] += 1
                elif by_result[i.id].outcome in ("failed", "error"):
                    row["failing"] += 1
        return sorted(rows.values(), key=lambda r: (-r["max_risk"], r["category"]))

    def coverage_by_risk(self) -> dict[int, dict[str, int]]:
        uncovered = {u.intent_id for u in self.uncovered}
        out: dict[int, dict[str, int]] = {}
        for i in self.intents:
            row = out.setdefault(i.risk, {"planned": 0, "covered": 0})
            row["planned"] += 1
            row["covered"] += int(i.id not in uncovered)
        return dict(sorted(out.items(), reverse=True))

    def product_bugs(self) -> list[Finding]:
        return sorted(
            [f for f in self.findings if f.classification == "product_bug"],
            key=lambda f: SEVERITY_ORDER[f.severity],
        )

    def blocking(self) -> list[Finding]:
        return [f for f in self.product_bugs() if f.severity in ("critical", "high")]

    def effort_estimate(self) -> dict[str, Any]:
        executed = sum(1 for r in self.results if r.outcome in ("passed", "failed"))
        bugs = len(self.product_bugs())
        minutes = executed * MINUTES_PER_TEST + bugs * MINUTES_PER_FINDING
        return {
            "executed_tests": executed,
            "product_findings": bugs,
            "minutes_per_test": MINUTES_PER_TEST,
            "minutes_per_finding": MINUTES_PER_FINDING,
            "hours": round(minutes / 60, 1),
            "formula": f"({executed} tests x {MINUTES_PER_TEST} min + {bugs} findings x {MINUTES_PER_FINDING} min) / 60",
        }


# ------------------------------------------------------------------ judge


def judge_report(
    report: RunReport, client: LLMClient, only_ambiguous: bool = True
) -> JudgeScores | None:
    """Score the report. With cheap verifiers first, only ambiguous product findings (low
    confidence or high/critical severity) are sent; otherwise every product finding is."""
    ambiguous = [
        f
        for f in report.findings
        if f.classification == "product_bug"
        and (not only_ambiguous or f.confidence < 0.7 or f.severity in ("critical", "high"))
    ]
    payload = {
        "summary": {"outcomes": report.outcome_counts(), "findings": len(report.findings)},
        "findings": [
            {
                "id": f.id,
                "title": f.title,
                "classification": f.classification,
                "severity": f.severity,
                "confidence": f.confidence,
                "root_cause_hypothesis": f.root_cause_hypothesis,
                "repro_curl": f.repro_curl,
                "evidence": [{"ref": e.ref, "excerpt": e.excerpt[:400]} for e in f.evidence],
            }
            for f in ambiguous
        ],
    }
    prompt = load_prompt("judge")
    try:
        with tracing.span(
            "judge",
            "agent",
            **{"agentqa.agent": "judge", "agentqa.findings_checked": len(ambiguous)},
        ):
            res = client.complete(
                prompt.render(payload=normalize(json.dumps(redact(payload), indent=1))),
                response_schema=JudgeScores,
                max_tokens=700,
                metadata=prompt.metadata("judge"),
            )
    except LLMError as exc:
        tracing.event("judge.failed", error=str(exc)[:200])
        return None
    scores = cast(JudgeScores, res.parsed)
    checked = {f.id for f in ambiguous}
    for f in report.findings:
        if f.id in checked:
            f.judge_checked = True
        if f.id in scores.unsupported_finding_ids and f.id in checked:
            f.classification = "needs_review"
            metrics.inc("agentqa_guardrail_events_total", guardrail="judge", action="downgrade")
    return scores


# ------------------------------------------------------------------ rendering

_env = Environment(
    autoescape=False, trim_blocks=True, lstrip_blocks=True
)  # Markdown (escaped at HTML render)
_html_env = Environment(
    autoescape=select_autoescape(["html"], default_for_string=True),
    trim_blocks=True,
    lstrip_blocks=True,
)

MD_TEMPLATE = """# AgentQA report — {{ r.api_title }} {{ r.api_version }}

Run `{{ r.run_id }}` · {{ r.created_at }} · profile `{{ r.profile }}` · strategy `{{ r.strategy }}` · target `{{ r.target }}`
{% if r.simulated %}

> **SIMULATED MODELS.** This run used the deterministic simulated models (ADR 0003). It
> demonstrates the pipeline's mechanics; it says nothing about a real model's quality.
{% endif %}
{% if r.aborted %}

> **Run aborted early:** {{ r.aborted }}. This is a partial report.
{% endif %}

Trace: `{{ r.trace_id or "n/a" }}`{% if r.phoenix_url %} · [open in Phoenix]({{ r.phoenix_url }}){% endif %}


## Outcome
| passed | failed | error | skipped | blocked |
|---|---|---|---|---|
| {{ o.passed }} | {{ o.failed }} | {{ o.error }} | {{ o.skipped }} | {{ o.blocked }} |

Findings: {{ r.findings|length }} ({{ r.product_bugs()|length }} product bugs, {{ r.blocking()|length }} blocking).
Hallucination rate (grounding violations before repair): {{ "%.1f"|format(r.hallucination.get("rate", 0) * 100) }}% of {{ r.hallucination.get("checked", 0)|int }} generated tests; {{ r.hallucination.get("quarantined", 0)|int }} quarantined.

## Findings
{% for f in findings %}
### {{ f.id }} · {{ f.severity|upper }} · {{ f.classification }} · {{ f.title }}
- Endpoint: `{{ f.endpoint }}`{% if f.endpoints|length > 1 %} (same root cause on {{ f.endpoints|length }} endpoints: {{ f.endpoints|join(", ") }}){% endif %} · category: {{ f.category }} · confidence {{ "%.2f"|format(f.confidence) }} · triaged by {{ f.triaged_by }}{% if f.judge_checked %} · judge-checked{% endif %}

- Root cause hypothesis: {{ f.root_cause_hypothesis }}
- Tests: {{ f.test_names|join(", ") }}
- Repro:
```bash
{{ f.repro_curl }}
```
- Evidence:
{% for e in f.evidence %}
  - `{{ e.ref }}` ({{ e.kind }}): {{ e.excerpt|replace("\\n", " ")|truncate(300) }}
{% endfor %}
{% else %}
No failing tests.
{% endfor %}

## Coverage by category
| category | max risk | planned | tested | passed | failing | uncovered |
|---|---|---|---|---|---|---|
{% for c in r.coverage() %}
| {{ c.category }} | {{ c.max_risk }} | {{ c.planned }} | {{ c.tested }} | {{ c.passed }} | {{ c.failing }} | {{ c.uncovered }} |
{% endfor %}

## Coverage by risk
| risk | planned | covered |
|---|---|---|
{% for risk, row in r.coverage_by_risk().items() %}
| {{ risk }} | {{ row.planned }} | {{ row.covered }} |
{% endfor %}

{% if r.uncovered %}
### Uncovered intents (never hidden)
| intent | endpoint | risk | reason |
|---|---|---|---|
{% for u in r.uncovered %}
| {{ u.intent_id }} | `{{ u.endpoint }}` | {{ u.risk }} | {{ u.reason }} |
{% endfor %}
{% endif %}

## Guardrails
- Quarantined document chunks (prompt injection): {{ r.quarantined_chunks|join(", ") or "none" }}
- Sandbox / guardrail events during execution: {{ r.guardrail_events|length }}
{% for e in r.guardrail_events[:20] %}
  - {{ e.get("guardrail", "") }} {{ e.get("action", "") }}: {{ e.get("kind", "") }} {{ e.get("detail", "") }} ({{ e.get("test", "") }})
{% endfor %}

## Cost and delegation
- Tokens: {{ r.cost.get("total_tokens", 0) }} ({{ r.cost.get("input_tokens", 0) }} in, {{ r.cost.get("output_tokens", 0) }} out, {{ r.cost.get("cached_input_tokens", 0) }} cached) in {{ r.cost.get("calls", 0) }} LLM calls
- Cost: ${{ "%.4f"|format(r.cost.get("cost_usd_actual", 0)) }} actual, ${{ "%.4f"|format(r.cost.get("cost_usd_list_equivalent", 0)) }} list-equivalent
- Tests by tier: {% for k, v in tier_counts.items() %}{{ k }}={{ v }} {% endfor %}

- Escalations: {{ r.delegation.get("escalations", 0) }}; estimated tokens saved vs strong-only: {{ r.delegation.get("tokens_saved_estimate", 0) }}
{% if r.incremental %}
- Incremental run: {{ r.incremental }}
{% endif %}
{% if r.lessons_used %}
- Lessons from earlier runs used in prompts: {{ r.lessons_used }}
{% endif %}

{% if r.judge %}
## Report quality (LLM judge, different model family)
accuracy {{ r.judge.accuracy }}/5 · actionability {{ r.judge.actionability }}/5 · clarity {{ r.judge.clarity }}/5{% if r.judge.unsupported_finding_ids %} · unsupported: {{ r.judge.unsupported_finding_ids|join(", ") }}{% endif %}

{% endif %}
## All tests
| test | intent | tier | outcome |
|---|---|---|---|
{% for t in r.results %}
| {{ t.test_name }} | {{ t.intent_id }} | {{ r.tiers.get(t.intent_id, "") }} | {{ t.outcome }}{% if t.flaky %} (flaky){% endif %} |
{% endfor %}
"""

EXEC_TEMPLATE = """# Executive summary — {{ r.api_title }}

{% if r.simulated %}
*This summary comes from a demonstration run with simulated AI models. The structure is real;
the numbers describe the demo, not your API.*

{% endif %}
**What we did.** We read your API specification and requirement documents, planned
{{ r.intents|length }} checks across {{ r.coverage()|length }} risk areas, and ran
{{ e.executed_tests }} automated tests against `{{ r.target }}`.

**What we found.** {% if r.blocking() %}{{ r.blocking()|length }} issue{{ "s" if r.blocking()|length != 1 }} should block the release{% else %}Nothing found blocks the release{% endif %}.
{{ r.product_bugs()|length }} confirmed product issue{{ "s" if r.product_bugs()|length != 1 }} in total{% if needs_review %}, and {{ needs_review }} item{{ "s" if needs_review != 1 }} need a person to review{% endif %}.

{% if r.blocking() %}
## Blocking the release
{% for f in r.blocking() %}
- **{{ f.title }}** ({{ f.severity }}): {{ f.root_cause_hypothesis }}
{% endfor %}

{% endif %}
## Findings by severity
| critical | high | medium | low |
|---|---|---|---|
| {{ sev.critical }} | {{ sev.high }} | {{ sev.medium }} | {{ sev.low }} |

## Coverage by risk area
| area | checks planned | tested | issues found | not covered |
|---|---|---|---|---|
{% for c in r.coverage() %}
| {{ c.category }} | {{ c.planned }} | {{ c.tested }} | {{ c.failing }} | {{ c.uncovered }} |
{% endfor %}
{% if r.uncovered %}

Some lower-risk checks were not run (budget or failed generation); each is listed with its reason
in the technical report.
{% endif %}

## Effort (estimate, not a measurement)
Writing and running these tests by hand would take roughly **{{ e.hours }} hours**.
Formula: {{ e.formula }}. Assumptions: {{ e.minutes_per_test }} minutes to design, write and run
one API test case; {{ e.minutes_per_finding }} minutes to investigate and write up one finding.
Your team's numbers may differ.

## Cost of this run
${{ "%.2f"|format(r.cost.get("cost_usd_actual", 0)) }} actual model spend
(${{ "%.2f"|format(r.cost.get("cost_usd_list_equivalent", 0)) }} at list prices).
"""

HTML_SHELL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ title }}</title>
<style>
:root { --bg:#fff; --fg:#1b1f24; --muted:#5b6470; --line:#e3e6ea; --accent:#2457d6; --bad:#c62828; --warn:#b26a00; }
@media (prefers-color-scheme: dark) { :root { --bg:#14171b; --fg:#e7eaee; --muted:#9aa4b0; --line:#2a3038; --accent:#7aa2ff; --bad:#ff7b7b; --warn:#ffb84d; } }
body { background:var(--bg); color:var(--fg); font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; margin:0 auto; max-width:1100px; padding:24px 16px; }
h1,h2,h3 { line-height:1.25 } h2 { border-bottom:1px solid var(--line); padding-bottom:4px; margin-top:32px }
table { border-collapse:collapse; width:100%; margin:8px 0; font-size:14px; display:block; overflow-x:auto }
th,td { border:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top }
code,pre { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px }
pre { background:color-mix(in srgb, var(--fg) 6%, transparent); padding:10px; overflow-x:auto; border-radius:6px }
blockquote { border-left:4px solid var(--warn); margin:12px 0; padding:4px 12px; color:var(--muted) }
a { color:var(--accent) }
</style></head><body>
{{ body|safe }}
</body></html>"""


def _md_to_html(md: str) -> str:
    """Small Markdown subset renderer (headings, tables, lists, code fences, quotes, bold, code)."""
    out: list[str] = []
    lines = md.splitlines()
    i = 0

    def inline(text: str) -> str:
        import re

        t = html.escape(text)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
        return t

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(html.escape(lines[i]))
                i += 1
            out.append("<pre><code>" + "\n".join(block) + "</code></pre>")
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not set(lines[i].replace("|", "").strip()) <= {"-", " "}:
                    rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            i -= 1
            if rows:
                head = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
                body = "".join(
                    "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:]
                )
                out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{inline(line[2:])}</blockquote>")
        elif line.lstrip().startswith("- "):
            items = []
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                items.append(f"<li>{inline(lines[i].lstrip()[2:])}</li>")
                i += 1
            i -= 1
            out.append("<ul>" + "".join(items) + "</ul>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
        i += 1
    return "\n".join(out)


def render(report: RunReport, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    findings = sorted(
        report.findings,
        key=lambda f: (f.classification != "product_bug", SEVERITY_ORDER[f.severity]),
    )
    tier_counts = dict(Counter(report.tiers.values()))
    md = _env.from_string(MD_TEMPLATE).render(
        r=report, o=report.outcome_counts(), findings=findings, tier_counts=tier_counts
    )
    sev: Counter[str] = Counter(f.severity for f in report.product_bugs())
    exec_md = _env.from_string(EXEC_TEMPLATE).render(
        r=report,
        e=report.effort_estimate(),
        sev={k: sev.get(k, 0) for k in SEVERITY_ORDER},
        needs_review=sum(f.classification == "needs_review" for f in report.findings),
    )
    md, exec_md = redact(md, pii=False), redact(exec_md, pii=False)
    paths = {
        "report_md": out_dir / "report.md",
        "report_html": out_dir / "report.html",
        "summary_md": out_dir / "executive_summary.md",
        "summary_html": out_dir / "executive_summary.html",
        "report_json": out_dir / "report.json",
    }
    paths["report_md"].write_text(md, encoding="utf-8")
    paths["summary_md"].write_text(exec_md, encoding="utf-8")
    shell = _html_env.from_string(HTML_SHELL)
    paths["report_html"].write_text(
        shell.render(title=f"AgentQA report {report.run_id}", body=_md_to_html(md)),
        encoding="utf-8",
    )
    paths["summary_html"].write_text(
        shell.render(title=f"Executive summary {report.run_id}", body=_md_to_html(exec_md)),
        encoding="utf-8",
    )
    paths["report_json"].write_text(report.model_dump_json(indent=1), encoding="utf-8")
    return paths
