"""Hosted mode (the public demo): private workspaces, guarded runs, the watch page and videos."""

from __future__ import annotations

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
        "SoftwareApplication",
        "VideoObject",
        "FAQPage",
    ]
    video = graph[2]
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
