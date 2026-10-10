"""Hosted mode (the public demo): private workspaces, guarded runs, the watch page and videos."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient
from streamlit.testing.v1 import AppTest

from agentqa.veroniqa import create_project, hosted, server
from agentqa.veroniqa.runs import NotRunnable, run_project_tests

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "agentqa/veroniqa/app.py")


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTQA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AGENTQA_CACHE_MODE", "off")
    monkeypatch.setenv("VERONIQA_HOSTED", "1")


def test_workspace_ids_are_checked() -> None:
    wid = hosted.new_workspace_id()
    assert hosted.workspace_root(wid).parent.name == wid
    for bad in ("../etc", "ABCDEF0123456789", "0123", "0123456789abcdef/../x"):
        with pytest.raises(ValueError):
            hosted.workspace_root(bad)


def test_prune_removes_idle_and_excess_workspaces(monkeypatch: pytest.MonkeyPatch) -> None:
    now = time.time()
    ids = [f"{i:016x}" for i in range(5)]
    for i, wid in enumerate(ids):
        hosted.touch(wid)
        age = hosted.WORKSPACE_TTL + 60 if i < 2 else i
        os.utime(hosted.workspace_root(wid).parent, (now - age, now - age))
    assert hosted.prune_workspaces(now=now, keep=ids[0]) == 1  # ids[1] idle; ids[0] is in use
    monkeypatch.setattr(hosted, "MAX_WORKSPACES", 2)
    hosted.prune_workspaces(now=now)
    left = sorted(p.name for p in hosted.workspaces_root().iterdir())
    assert left == [ids[2], ids[3]]  # the two most recently used


def test_hosted_runs_only_target_the_bundled_api() -> None:
    root = hosted.workspace_root(hosted.new_workspace_id())
    mine = create_project("My API", root=root, base_url="https://api.example.com", spec="x.json")
    with pytest.raises(NotRunnable, match="public demo"):
        run_project_tests(mine, "premium")


def _wid(at: AppTest) -> str:
    value = at.query_params["w"]
    return value if isinstance(value, str) else value[0]


def test_hosted_app_gives_each_visitor_a_demo_workspace() -> None:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    wid = _wid(at)
    assert hosted.WORKSPACE_ID.match(wid)
    assert at.header[0].value == "Orders API demo"
    assert not at.selectbox or all(s.label != "Model profile" for s in at.selectbox)
    assert any("public demo" in i.value for i in at.info)  # settings are locked
    assert not any(t.label.startswith("Base URL") for t in at.text_input)

    other = AppTest.from_file(APP, default_timeout=120)
    other.run()
    assert _wid(other) != wid  # a second visitor gets their own workspace

    again = AppTest.from_file(APP, default_timeout=120)
    again.query_params["w"] = wid
    again.run()
    assert _wid(again) == wid  # the link brings the same workspace back


def test_home_page_is_the_introduction_and_the_app_is_at_talk() -> None:
    client = TestClient(Starlette(routes=server.app._user_routes))
    page = client.get("/")
    assert page.status_code == 200 and "Content-Security-Policy" in page.headers
    assert (
        '<video src="/videos/veroniqa-intro.mp4" poster="/videos/veroniqa-intro.jpg"' in page.text
    )
    assert page.text.count("<video") == 1 and 'href="/talk/"' in page.text
    for old, new in (("/watch", "/"), ("/talk", "/talk/")):
        moved = client.get(old, follow_redirects=False)
        assert moved.status_code == 308 and moved.headers["location"] == new
    part = client.get("/videos/veroniqa-intro.mp4", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100
    assert client.get("/videos/veroniqa-intro.jpg").headers["content-type"] == "image/jpeg"
    for other in ("01-cost-vs-quality.mp4", "veroniqa-intro.srt", "../pyproject.toml"):
        assert client.get(f"/videos/{other}").status_code == 404


def test_streamlit_is_served_under_talk() -> None:
    import streamlit as st

    assert st.config.get_option("server.baseUrlPath") == server.TALK == "talk"


def test_home_page_is_a_full_page_with_seo_metadata() -> None:
    import json
    import re

    client = TestClient(Starlette(routes=server.app._user_routes))
    page = client.get("/").text
    for part in (
        '<header class="site">',
        '<footer class="site">',
        '<main id="main">',
        '<link rel="canonical" href="https://veroniqa.vercel.app/">',
        'property="og:image"',
        'name="twitter:card"',
        'name="description"',
        'href="/talk/"',
        'id="transcript"',
    ):
        assert part in page, part
    assert page.count("<h1") == 1
    ld = re.search(r'<script type="application/ld\+json">(.*?)</script>', page, re.S)
    assert ld
    graph = json.loads(ld.group(1))["@graph"]
    assert [g["@type"] for g in graph] == [
        "WebSite",
        "Person",
        "SoftwareApplication",
        "VideoObject",
        "FAQPage",
    ]
    video = next(g for g in graph if g["@type"] == "VideoObject")
    assert video["duration"] == "PT1M25S" and "Meet VeroniQA" in video["transcript"]


def test_robots_sitemap_llms_and_favicon() -> None:
    client = TestClient(Starlette(routes=server.app._user_routes))
    robots = client.get("/robots.txt").text
    assert "User-agent: GPTBot\nAllow: /" in robots and "User-agent: ClaudeBot" in robots
    assert "Sitemap: https://veroniqa.vercel.app/sitemap.xml" in robots
    sitemap = client.get("/sitemap.xml")
    assert sitemap.headers["content-type"].startswith("application/xml")
    assert "<video:content_loc>https://veroniqa.vercel.app/videos/veroniqa-intro.mp4" in (
        sitemap.text
    )
    llms = client.get("/llms.txt").text
    assert llms.startswith("# VeroniQA\n\n> ") and "simulated models" in llms
    assert client.get("/favicon.svg").headers["content-type"] == "image/svg+xml"


def test_hosted_visitor_can_create_a_project() -> None:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert at.header[0].value == "Orders API demo"
    next(t for t in at.text_input if t.label == "Name").set_value("Billing API")
    next(b for b in at.button if b.label == "Create project").click().run()
    assert not at.exception, at.exception
    assert at.header[0].value == "Billing API"
    assert (hosted.workspace_root(_wid(at)) / "billing-api" / "project.yaml").exists()


def test_home_page_stats_match_the_result_files() -> None:
    import json

    from agentqa.evals.report import latest_results
    from agentqa.veroniqa import site

    func_path, perf_path = latest_results()
    assert func_path and perf_path
    st = json.loads(func_path.read_text())["strategies"]
    cases = json.loads(perf_path.read_text())["cases"]
    s0, s3 = st["S0"]["aggregate"], st["S3"]["aggregate"]
    ratio = s3["cost_usd_list_equivalent"]["mean"] / s0["cost_usd_list_equivalent"]["mean"]
    assert site.STATS["recall_pct"] == round(s3["recall"]["mean"] * 100)
    assert site.STATS["cost_pct"] == round(ratio * 100)
    assert site.STATS["perf_caught"] == sum(bool(c["regressed"]) for c in cases)
    assert site.STATS["perf_total"] == len(cases)
    assert site.STATS["cost_levers"] == sum(name.startswith("S3-no-") for name in st)
    from agentqa.evals.groundtruth import BUG_IDS

    assert site.STATS["planted_bugs"] == len(BUG_IDS)
    page = site.home_page()
    for _icon, value, caption in site.stat_tiles():
        assert f">{value}</b><span>{caption}</span>" in page


def _agent_client() -> TestClient:
    """The site's routes, 404 handler and /talk/ middleware, with a stand-in for Streamlit's
    app shell (the real one needs a running Streamlit runtime)."""
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, PlainTextResponse, Response
    from starlette.routing import Route

    shell = "<!doctype html><html><head><title>Streamlit</title></head><body><div id=root></div></body></html>"

    async def talk(request: Request) -> Response:
        return HTMLResponse(shell)

    async def health(request: Request) -> Response:
        return PlainTextResponse("ok")

    routes = [
        *server.app._user_routes,
        Route("/talk/", talk),
        Route("/talk/_stcore/health", health),
    ]
    return TestClient(
        Starlette(
            routes=routes,
            middleware=[Middleware(server.AgentFriendlyTalk)],
            exception_handlers={404: server.not_found},
        )
    )


@pytest.mark.parametrize("path", ["/", "/privacy"])
def test_pages_negotiate_markdown_with_vary_and_link(path: str) -> None:
    client = _agent_client()
    page = client.get(path, headers={"Accept": "text/html"})
    md = client.get(path, headers={"Accept": "text/markdown"})
    assert page.headers["content-type"].startswith("text/html")
    assert md.headers["content-type"] == "text/markdown; charset=utf-8"
    assert md.text.startswith("# ")
    for r in (page, md):
        assert r.headers["vary"] == "Accept"  # one header, one value
        assert 'rel="alternate"; type="text/markdown"' in r.headers["link"]
    twin = client.get(page.headers["link"].split(">")[0][1:])
    assert twin.status_code == 200 and twin.text == md.text


def test_unknown_addresses_are_real_404s_with_a_way_on() -> None:
    client = _agent_client()
    for path in ("/nope", "/talk/nope", "/talk/a/b"):
        page = client.get(path, headers={"Accept": "text/html"})
        md = client.get(path, headers={"Accept": "text/markdown"})
        assert page.status_code == md.status_code == 404, path
        assert "Page not found" in page.text and 'href="/llms.txt"' in page.text
        assert md.text.startswith("# Page not found (404)") and "/llms.txt" in md.text
        assert md.headers["content-type"] == "text/markdown; charset=utf-8"
    assert client.get("/talk/_stcore/health").text == "ok"  # Streamlit's own paths still work


def test_talk_shell_is_readable_without_javascript_and_as_markdown() -> None:
    client = _agent_client()
    page = client.get("/talk/", headers={"Accept": "text/html"})
    assert "<title>Talk to VeroniQA" in page.text and "<noscript><main><h1>" in page.text
    assert 'name="description"' in page.text and int(page.headers["content-length"]) == len(
        page.content
    )
    assert page.headers["vary"] == "Accept"
    head = client.head("/talk/")
    assert (
        head.status_code == 200 and head.headers["content-length"] == page.headers["content-length"]
    )
    md = client.get("/talk/", headers={"Accept": "text/markdown"})
    assert md.text.startswith("# Talk to VeroniQA") and "no sign-up" in md.text.lower()


def test_trust_files_and_identity() -> None:
    from datetime import UTC, datetime

    from agentqa.veroniqa import site

    client = _agent_client()
    sec = client.get("/.well-known/security.txt").text
    assert sec.startswith("Contact: https://") and "Expires: " in sec
    expires = datetime.fromisoformat(sec.split("Expires: ")[1].split()[0].replace("Z", "+00:00"))
    assert 0 < (expires - datetime.now(UTC)).days <= 365  # RFC 9116: under a year
    assert client.get("/site.webmanifest").json()["name"].startswith("VeroniQA")
    assert "Content-Signal: search=yes" in client.get("/robots.txt").text
    assert "/privacy</loc>" in client.get("/sitemap.xml").text
    home = client.get("/").text
    for part in ('name="author"', 'rel="manifest"', 'href="/privacy"', "security.txt"):
        assert part in home
    graph = json.loads(site.home_page().split('application/ld+json">')[1].split("</script>")[0])
    assert any(n["@type"] == "Person" and n["url"] == site.AUTHOR_URL for n in graph["@graph"])


@pytest.mark.parametrize(
    ("accept", "markdown"),
    [
        ("text/markdown", True),
        ("text/markdown, text/html;q=0.9", True),
        ("text/html, text/markdown;q=0.5", False),
        ("text/html,application/xhtml+xml,*/*;q=0.8", False),
        ("*/*", False),
        ("", False),
    ],
)
def test_prefers_markdown(accept: str, markdown: bool) -> None:
    from agentqa.veroniqa import site

    assert site.prefers_markdown(accept) is markdown
