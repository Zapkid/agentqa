"""What keeps the hosted site's function CPU low: CDN-cacheable responses, lazy Chroma, and a guard
that stops an unattended /talk/ tab from reconnecting forever."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient

from agentqa.veroniqa import idle_guard, server

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTQA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AGENTQA_CACHE_MODE", "off")
    monkeypatch.setenv("VERONIQA_HOSTED", "1")


def _client() -> TestClient:
    shell = "<!doctype html><html><head><title>Streamlit</title></head><body><div id=root></div></body></html>"

    async def talk(request: Request) -> Response:
        return HTMLResponse(shell)

    async def asset(request: Request) -> Response:  # what Streamlit sends for a hashed file
        return PlainTextResponse("x", headers={"Cache-Control": "public, max-age=31536000"})

    async def health(
        request: Request,
    ) -> Response:  # Streamlit gives each visitor an XSRF cookie here
        return PlainTextResponse(
            "ok", headers={"Cache-Control": "no-cache", "Set-Cookie": "_streamlit_xsrf=abc; Path=/"}
        )

    async def host_config(request: Request) -> Response:
        return JSONResponse({"allowedOrigins": []}, headers={"Cache-Control": "no-cache"})

    async def missing(request: Request) -> Response:
        return PlainTextResponse("no", status_code=404)

    routes = [
        *server.app._user_routes,
        Route("/talk/", talk),
        Route("/talk/static/js/index.abc123.js", asset, methods=["GET", "HEAD", "POST"]),
        Route("/talk/static/js/gone.js", missing),
        Route("/talk/_stcore/health", health),
        Route("/talk/_stcore/host-config", host_config),
    ]
    return TestClient(
        Starlette(
            routes=routes,
            middleware=[Middleware(server.AgentFriendlyTalk)],
            exception_handlers={404: server.not_found},
        )
    )


def _edge_seconds(headers: dict[str, str]) -> int | None:
    """The CDN lifetime a response asks for (s-maxage), or None when the CDN may not cache it."""
    for part in headers.get("cache-control", "").split(","):
        name, _, value = part.strip().partition("=")
        if name == "s-maxage":
            return int(value)
    return None


@pytest.mark.parametrize("path", ["/", "/privacy", "/developers"])
def test_pages_are_cacheable_by_the_cdn_and_vary_on_accept(path: str) -> None:
    client = _client()
    for accept in ("text/html", "text/markdown"):
        r = client.get(path, headers={"Accept": accept})
        assert r.status_code == 200 and _edge_seconds(r.headers) == 3600, (path, accept)
        assert r.headers["vary"] == "Accept"  # so HTML and Markdown are cached separately
        assert "set-cookie" not in r.headers  # a cookie would stop the CDN caching it


@pytest.mark.parametrize(
    "path",
    [
        "/index.md",
        "/privacy.md",
        "/developers.md",
        "/talk.md",
        "/robots.txt",
        "/sitemap.xml",
        "/llms.txt",
        "/.well-known/security.txt",
        "/.well-known/api-catalog",
        "/site.webmanifest",
        "/favicon.svg",
        "/openapi.json",
        "/openapi.yaml",
        "/api/openapi.json",
    ],
)
def test_files_are_cacheable_by_the_cdn(path: str) -> None:
    r = _client().get(path)
    assert r.status_code == 200 and _edge_seconds(r.headers) == 3600, path
    assert "max-age=86400" in r.headers["cache-control"]  # browsers keep their day


def test_video_and_poster_are_cached_for_a_day() -> None:
    client = _client()
    for path in ("/videos/veroniqa-intro.mp4", "/videos/veroniqa-intro.jpg"):
        r = client.get(path, headers={"Range": "bytes=0-9"})
        assert r.status_code == 206 and _edge_seconds(r.headers) == 86400, path


def test_not_found_is_cached_briefly_and_the_api_is_not_cached() -> None:
    client = _client()
    missing = client.get("/no-such-page", headers={"Accept": "text/html"})
    assert missing.status_code == 404 and _edge_seconds(missing.headers) == 600
    for path in ("/api/v1/status", "/api/v1/results", "/api/v1/nope"):
        r = client.get(path)
        assert _edge_seconds(r.headers) is None, path  # per-client rate-limit headers
        assert r.headers["cache-control"] == "no-store"


def test_streamlit_assets_are_cacheable_but_errors_and_other_calls_are_not() -> None:
    client = _client()
    asset = client.get("/talk/static/js/index.abc123.js")
    assert asset.status_code == 200 and _edge_seconds(asset.headers) == 86400
    # Streamlit's own header is replaced, not duplicated
    assert asset.headers["cache-control"] == "public, max-age=86400, s-maxage=86400"
    assert _edge_seconds(client.head("/talk/static/js/index.abc123.js").headers) == 86400
    assert _edge_seconds(client.post("/talk/static/js/index.abc123.js").headers) is None
    gone = client.get("/talk/static/js/gone.js")
    assert gone.status_code == 404 and _edge_seconds(gone.headers) is None


def test_host_config_is_cacheable_but_the_health_check_never_is() -> None:
    client = _client()
    config = client.get("/talk/_stcore/host-config")
    assert config.status_code == 200 and _edge_seconds(config.headers) == 3600
    assert "set-cookie" not in config.headers
    # /_stcore/health hands each visitor their own XSRF cookie: caching it would share that cookie
    health = client.get("/talk/_stcore/health")
    assert (
        _edge_seconds(health.headers) is None and "_streamlit_xsrf" in health.headers["set-cookie"]
    )
    assert health.headers["cache-control"] == "no-cache"


def test_a_response_that_sets_a_cookie_is_never_made_cacheable() -> None:
    sent: list[dict] = []

    async def collect(message: dict) -> None:
        sent.append(message)

    import asyncio

    start = {
        "type": "http.response.start",
        "status": 200,
        "headers": [(b"set-cookie", b"a=b"), (b"cache-control", b"no-cache")],
    }
    asyncio.run(server._edge_cached(collect, "public, s-maxage=60")(start))
    assert dict(sent[0]["headers"])[b"cache-control"] == b"no-cache"  # untouched


def test_chroma_is_not_loaded_until_a_store_is_opened() -> None:
    code = (
        "import sys, agentqa.veroniqa.server\n"
        "assert 'chromadb' not in sys.modules, 'chromadb imported at startup'\n"
        "from agentqa.ingest.vectorstore import VectorStore\n"
        "VectorStore()\n"
        "assert 'chromadb' in sys.modules\n"
    )
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, done.stderr[-600:]


# ------------------------------------------------------------------ the idle-tab guard


def test_app_page_carries_the_idle_guard_once_and_before_streamlit() -> None:
    page = _client().get("/talk/", headers={"Accept": "text/html"}).text
    assert page.count("<script>(function () {") == 1
    assert page.index("Paused while you were away") < page.index("</head>")
    assert idle_guard.SCRIPT in page and "__IDLE_MS__" not in page
    assert f"var IDLE_MS = {idle_guard.IDLE_MS};" in page and idle_guard.IDLE_MS == 120_000


def test_idle_guard_only_pauses_streamlits_own_requests_when_away() -> None:
    js = idle_guard.SCRIPT
    # pauses only after the app's WebSocket closes while the tab is hidden or idle
    assert "'/_stcore/stream'" in js and "document.hidden || Date.now() - lastInput > IDLE_MS" in js
    # holds back only Streamlit's own calls (the retry loop's health and host-config checks)
    assert "paused && url.indexOf('/_stcore/') !== -1" in js
    # a person coming back (tab visible, click, key, scroll, touch) restarts the session cleanly
    for event in ("visibilitychange", "pointerdown", "keydown", "wheel", "touchstart", "scroll"):
        assert event in js
    assert "window.location.reload()" in js
    assert "role', 'status'" in js and "id = 'vq-paused'" in js  # a visible, accessible notice


def test_the_app_shell_is_cacheable_in_both_forms_but_not_when_it_sets_a_cookie() -> None:
    client = _client()
    for accept in ("text/html", "text/markdown"):
        r = client.get("/talk/", headers={"Accept": accept})
        assert r.status_code == 200 and _edge_seconds(r.headers) == 3600, accept
        assert r.headers["vary"] == "Accept" and "set-cookie" not in r.headers
    assert _edge_seconds(client.head("/talk/").headers) == 3600
    # if Streamlit ever put a per-visitor cookie on the shell, it must not be shared
    cookie = _client_with_shell_cookie().get("/talk/", headers={"Accept": "text/html"})
    assert _edge_seconds(cookie.headers) is None and "set-cookie" in cookie.headers


def _client_with_shell_cookie() -> TestClient:
    async def talk(request: Request) -> Response:
        return HTMLResponse(
            "<!doctype html><html><head><title>S</title></head><body></body></html>",
            headers={"Set-Cookie": "visitor=1; Path=/"},
        )

    return TestClient(
        Starlette(routes=[Route("/talk/", talk)], middleware=[Middleware(server.AgentFriendlyTalk)])
    )
