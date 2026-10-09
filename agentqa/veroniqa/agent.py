"""Veroniqa: routes a chat message to one action and carries it out with the repo's features."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from agentqa.guards import injection
from agentqa.llm.prompts import load_prompt
from agentqa.llm.router import ModelRouter
from agentqa.llm.types import LLMError
from agentqa.obs import tracing
from agentqa.veroniqa import rag
from agentqa.veroniqa.fetch import FetchedPage, FetchError, fetch_url
from agentqa.veroniqa.knowledge import KnowledgeBase, Passage
from agentqa.veroniqa.projects import Project
from agentqa.veroniqa.runs import NotRunnable, project_runs, run_project_tests

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
RUN_ID_RE = re.compile(r"\brun-\d{8}-\d{6}-[0-9a-f]{6}\b")

Action = Literal["ask", "add_link", "run_tests", "list_runs", "show_report", "list_sources", "help"]

HELP = """I'm **Veroniqa**. In this project I can:

- **answer questions** from the documents and links in its knowledge base, citing the passages I used;
- **add a link**: send me a web page URL and I'll read it into the knowledge base;
- **run the tests**: plan, generate and run API tests against the project's target, then triage the failures (for the demo project, say which seeded bugs to switch on, e.g. "run the tests with B01 and B04");
- **list runs**, and **show a report** (the latest, or a run id).

Upload documents in the Knowledge tab."""


class Route(BaseModel):
    action: Action
    argument: str = Field(default="", description="question, URL or run id, depending on action")
    bugs: str = Field(default="", description='demo project only: "all", "clean" or "B01,B04"')


class Reply(BaseModel):
    text: str
    action: str
    citations: list[Passage] = Field(default_factory=list)
    run_id: str | None = None
    report_html: str | None = None
    note: str = ""


class Veroniqa:
    def __init__(
        self,
        project: Project,
        *,
        profile: str | None = None,
        router: ModelRouter | None = None,
        fetcher: Callable[[str], FetchedPage] = fetch_url,
        runner: Callable[[Project, str, str], Any] = run_project_tests,
    ) -> None:
        self.project = project
        self.profile = profile or os.environ.get("AGENTQA_PROFILE", "simulated")
        self.router = router or ModelRouter(self.profile)
        self.router_client = self.router.for_tier("T1")  # routing is easy: cheapest tier
        self.client = self.router.for_role("veroniqa")
        self.kb = KnowledgeBase(
            project,
            classifier=injection.llm_classifier(self.router.for_role("injection_classifier")),
        )
        self.fetcher = fetcher
        self.runner = runner

    # ------------------------------------------------------------------ entry point

    def chat(self, message: str) -> Reply:
        message = message.strip()[:4000]
        if not message:
            return Reply(text=HELP, action="help")
        with tracing.span("veroniqa.chat", "agent", **{"agentqa.agent": "veroniqa"}):
            route = self.route(message)
            handler: dict[str, Callable[[str, Route], Reply]] = {
                "ask": self._ask,
                "add_link": self._add_link,
                "run_tests": self._run_tests,
                "list_runs": self._list_runs,
                "show_report": self._show_report,
                "list_sources": self._list_sources,
                "help": lambda _m, _r: Reply(text=HELP, action="help"),
            }
            return handler[route.action](message, route)

    def route(self, message: str) -> Route:
        prompt = load_prompt("veroniqa_route")
        cfg = self.project.config
        payload = {
            "message": message,
            "project": {
                "demo": cfg.local_demo,
                "has_spec": bool(cfg.spec),
                "has_target": bool(cfg.base_url) or cfg.local_demo,
                "sources": len(self.project.sources()),
            },
        }
        try:
            res = self.router_client.complete(
                prompt.render(payload=json.dumps(payload, indent=1)),
                response_schema=Route,
                max_tokens=200,
                metadata=prompt.metadata("veroniqa"),
            )
            return cast(Route, res.parsed)
        except LLMError:
            return Route(action="ask", argument=message)

    # ------------------------------------------------------------------ actions

    def ask(self, question: str) -> rag.Answer:
        return rag.answer(question, self.kb.retrieve(question), self.client)

    def _ask(self, message: str, route: Route) -> Reply:
        ans = self.ask(route.argument or message)
        return Reply(text=ans.text, action="ask", citations=ans.citations, note=ans.note)

    def _add_link(self, message: str, route: Route) -> Reply:
        found = URL_RE.findall(route.argument) or URL_RE.findall(message)
        if not found:
            return Reply(
                text="Send me the full link, starting with http:// or https://.", action="add_link"
            )
        try:
            src = self.kb.add_link(found[0], self.fetcher)
        except (FetchError, ValueError) as exc:
            return Reply(text=f"I could not add that link: {exc}", action="add_link")
        extra = (
            f", {src.quarantined} quarantined as possible prompt injection"
            if src.quarantined
            else ""
        )
        return Reply(
            text=f"Added **{src.name}** to the knowledge base ({src.chunks} passages{extra}).",
            action="add_link",
        )

    def _run_tests(self, message: str, route: Route) -> Reply:
        try:
            out = self.runner(self.project, self.profile, route.bugs)
        except NotRunnable as exc:
            return Reply(text=str(exc), action="run_tests")
        r = out.report
        bugs = r.product_bugs()
        lines = [
            f"Run **{r.run_id}** finished: {r.outcome_counts()}.",
            f"{len(r.findings)} findings, {len(bugs)} product bugs, "
            f"{r.cost.get('total_tokens', 0)} tokens.",
        ]
        if r.simulated:
            lines.append(
                "_Simulated models: this shows the mechanics, not a real model's quality._"
            )
        for f in sorted(bugs, key=lambda f: f.severity)[:6]:
            lines.append(f"- **{f.severity}** {f.title} (`{f.endpoint}`)")
        return Reply(
            text="\n".join(lines),
            action="run_tests",
            run_id=r.run_id,
            report_html=str(out.paths["report_html"]),
        )

    def _list_runs(self, message: str, route: Route) -> Reply:
        runs = project_runs(self.project)
        if not runs:
            return Reply(text="No runs yet. Ask me to run the tests.", action="list_runs")
        rows = ["| run | when | findings | product bugs | tokens |", "|---|---|---|---|---|"]
        rows += [
            f"| {s.run_id} | {s.created_at} | {s.findings} | {s.product_bugs} | {s.tokens} |"
            for s in runs[:10]
        ]
        return Reply(text="\n".join(rows), action="list_runs")

    def _show_report(self, message: str, route: Route) -> Reply:
        runs = project_runs(self.project)
        wanted = RUN_ID_RE.findall(route.argument) or RUN_ID_RE.findall(message)
        run = next((s for s in runs if s.run_id in wanted), None) if wanted else None
        run = run or (runs[0] if runs and not wanted else None)
        if run is None:
            return Reply(text="I don't have that run. Ask me to list runs.", action="show_report")
        summary = Path(run.summary_md)
        text = summary.read_text(encoding="utf-8") if summary.exists() else "(no summary)"
        return Reply(
            text=text, action="show_report", run_id=run.run_id, report_html=run.report_html
        )

    def _list_sources(self, message: str, route: Route) -> Reply:
        sources = self.project.sources()
        if not sources:
            return Reply(
                text="The knowledge base is empty. Upload documents or send me a link.",
                action="list_sources",
            )
        rows = [
            f"- **{s.name}** ({s.kind}, {s.chunks} passages"
            + (f", {s.quarantined} quarantined" if s.quarantined else "")
            + ")"
            for s in sources
        ]
        return Reply(text="\n".join(rows), action="list_sources")
