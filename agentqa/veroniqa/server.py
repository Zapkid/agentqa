"""VeroniQA as one ASGI app: a home page with the introduction video, and the Streamlit UI at
/talk/.

    uvicorn agentqa.veroniqa.server:app --host 0.0.0.0 --port 8000

This is how the hosted demo runs (Dockerfile.vercel). Locally, `make veroniqa` is simpler. The
pages themselves are in site.py.

Agents get the same content as people: every page has a Markdown twin (`Accept: text/markdown`
or the `.md` address), unknown addresses return a real 404 with links onward, and the JavaScript
app at /talk/ carries a server-rendered description for clients that do not run scripts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import streamlit as st
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agentqa.veroniqa import site

APP = Path(__file__).with_name("app.py")
INTRO = site.INTRO  # captions are burned in
POSTER = INTRO.with_suffix(".jpg")
TALK = "talk"  # Streamlit's base path: the app, its assets and its WebSocket live under /talk/

# Must be set before st.App builds its routes. Plain `streamlit run` (make veroniqa) is unaffected.
st.config.set_option("server.baseUrlPath", TALK)

# The first path segments Streamlit serves under /talk/. Anything else would get the app shell
# with a 200 (a "soft 404"), so the middleware answers it with a real 404 instead.
STREAMLIT_SEGMENTS = frozenset(
    {"", "favicon.png", "manifest.json", "static", "_stcore", "media", "component", "auth", "app"}
)

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; media-src 'self'; style-src 'unsafe-inline'; img-src 'self'; "
        "manifest-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}
PAGE = {**SECURITY_HEADERS, "Cache-Control": "public, max-age=300", "Vary": "Accept"}
CACHE = {"Cache-Control": "public, max-age=86400"}
MARKDOWN = "text/markdown; charset=utf-8"


def _negotiated(
    request: Request, html: str, markdown: str, md_path: str, status: int = 200
) -> Response:
    """The same page as HTML or Markdown, by the Accept header, with a Link to the Markdown twin."""
    headers = {**PAGE, "Link": f'<{md_path}>; rel="alternate"; type="text/markdown"'}
    if site.prefers_markdown(request.headers.get("accept", "")):
        return Response(markdown, status_code=status, media_type=MARKDOWN, headers=headers)
    return HTMLResponse(html, status_code=status, headers=headers)


def _markdown(text: str, status: int = 200) -> Response:
    return Response(text, status_code=status, media_type=MARKDOWN, headers=CACHE)


async def home(request: Request) -> Response:
    return _negotiated(request, site.home_page(), site.home_markdown(), "/index.md")


async def home_md(request: Request) -> Response:
    return _markdown(site.home_markdown())


async def privacy(request: Request) -> Response:
    return _negotiated(request, site.privacy_page(), site.privacy_markdown(), "/privacy.md")


async def privacy_md(request: Request) -> Response:
    return _markdown(site.privacy_markdown())


async def talk_md(request: Request) -> Response:
    return _markdown(site.talk_markdown())


async def robots(request: Request) -> Response:
    return PlainTextResponse(site.robots_txt(), headers=CACHE)


async def sitemap(request: Request) -> Response:
    return Response(site.sitemap_xml(), media_type="application/xml", headers=CACHE)


async def llms(request: Request) -> Response:
    return PlainTextResponse(site.llms_txt(), headers=CACHE)


async def security(request: Request) -> Response:
    return PlainTextResponse(site.security_txt(), headers=CACHE)


async def manifest(request: Request) -> Response:
    return Response(site.web_manifest(), media_type="application/manifest+json", headers=CACHE)


async def favicon(request: Request) -> Response:
    return Response(site.FAVICON, media_type="image/svg+xml", headers=CACHE)


async def to_home(request: Request) -> Response:
    return RedirectResponse("/", status_code=308)


async def to_talk(request: Request) -> Response:
    return RedirectResponse(f"/{TALK}/", status_code=308)


async def intro(request: Request) -> Response:
    return FileResponse(INTRO, media_type="video/mp4", headers=CACHE)


async def poster(request: Request) -> Response:
    return FileResponse(POSTER, media_type="image/jpeg", headers=CACHE)


def not_found_response(request: Request) -> Response:
    path = request.url.path
    return _negotiated(
        request, site.not_found_page(path), site.not_found_markdown(path), "/index.md", status=404
    )


async def not_found(request: Request, exc: Exception) -> Response:
    return not_found_response(request)


class AgentFriendlyTalk:
    """ASGI middleware for /talk/: a Markdown description for agents that ask for it, a real 404
    for addresses Streamlit does not serve, and a server-rendered description, title and
    Markdown link in the app shell for clients that do not run JavaScript."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if scope["type"] != "http" or not path.startswith(f"/{TALK}/"):
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        segment = path[len(TALK) + 2 :].split("/", 1)[0]
        if segment not in STREAMLIT_SEGMENTS:
            await not_found_response(request)(scope, receive, send)
            return
        if path != f"/{TALK}/" or scope["method"] not in ("GET", "HEAD"):
            await self.app(scope, receive, send)
            return
        link = f'</{TALK}/>; rel="alternate"; type="text/markdown"'
        if site.prefers_markdown(request.headers.get("accept", "")):
            md = Response(
                site.talk_markdown(),
                media_type=MARKDOWN,
                headers={"Vary": "Accept", "Link": link, "Cache-Control": "no-cache"},
            )
            await md(scope, receive, send)
            return
        await self._shell(scope, receive, send, link)

    async def _shell(self, scope: Scope, receive: Receive, send: Send, link: str) -> None:
        # Ask for the shell uncompressed so it can be edited; the edge compresses it again.
        headers = [(k, v) for k, v in scope["headers"] if k.lower() != b"accept-encoding"]
        head_only = scope["method"] == "HEAD"
        scope = {**scope, "headers": headers, "method": "GET"}  # HEAD: same headers, no body
        start: dict[str, Any] = {}
        body = bytearray()

        async def capture(message: Message) -> None:
            if message["type"] == "http.response.start":
                start.update(message)
                return
            body.extend(message.get("body", b""))
            if message.get("more_body"):
                return
            out = bytes(body)
            ctype = dict(start.get("headers", [])).get(b"content-type", b"")
            if start.get("status") == 200 and ctype.startswith(b"text/html"):
                out = enrich_shell(out.decode("utf-8")).encode("utf-8")
            kept = [
                (k, v)
                for k, v in start.get("headers", [])
                if k.lower() not in (b"content-length", b"vary")
            ]
            kept += [
                (b"content-length", str(len(out)).encode()),
                (b"vary", b"Accept"),
                (b"link", link.encode()),
            ]
            await send({**start, "headers": kept})
            await send({"type": "http.response.body", "body": b"" if head_only else out})

        await self.app(scope, receive, capture)


def enrich_shell(page: str) -> str:
    """Give Streamlit's empty app shell a real title, description and readable content."""
    title = "Talk to VeroniQA · the AI assistant for API testing"
    description = (
        "The VeroniQA app: chat with cited answers, a knowledge base for your API documents, and "
        "AgentQA test runs. No sign-up, no API keys."
    )
    head = (
        f'<meta name="description" content="{description}">'
        f'<link rel="canonical" href="{site.site_url()}/{TALK}/">'
        f'<link rel="alternate" type="text/markdown" href="/{TALK}/">'
    )
    noscript = (
        "<noscript><main><h1>Talk to VeroniQA</h1>"
        "<p>This is an interactive app and needs JavaScript. VeroniQA is an AI assistant for API "
        "testing: ask about an API and get answers with sources, add documents and links "
        "(including OpenAPI and Swagger files) to a knowledge base, and run AgentQA's tests "
        "against a demo API. There is no sign-up and no API key.</p>"
        '<p><a href="/">Home page</a> · <a href="/index.md">Home page as Markdown</a> · '
        '<a href="/llms.txt">llms.txt</a></p></main></noscript>'
    )
    page = re.sub(r"<title>.*?</title>", f"<title>{title}</title>", page, count=1, flags=re.S)
    page = page.replace("</head>", head + "</head>", 1)
    return re.sub(r"(<body[^>]*>)", r"\1" + noscript.replace("\\", "\\\\"), page, count=1)


app = st.App(
    APP,
    routes=[
        Route("/", home),
        Route("/index.md", home_md),
        Route("/privacy", privacy),
        Route("/privacy.md", privacy_md),
        Route(f"/{TALK}.md", talk_md),
        Route("/watch", to_home),  # the home page's earlier address
        Route(f"/{TALK}", to_talk),
        Route("/robots.txt", robots),
        Route("/sitemap.xml", sitemap),
        Route("/llms.txt", llms),
        Route("/.well-known/security.txt", security),
        Route("/site.webmanifest", manifest),
        Route("/favicon.svg", favicon),
        Route("/videos/veroniqa-intro.mp4", intro),
        Route("/videos/veroniqa-intro.jpg", poster),
    ],
    middleware=[Middleware(AgentFriendlyTalk)],
    exception_handlers={404: not_found},
)
