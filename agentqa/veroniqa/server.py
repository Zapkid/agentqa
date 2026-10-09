"""VeroniQA as one ASGI app: a home page with the introduction video, and the Streamlit UI at
/talk/.

    uvicorn agentqa.veroniqa.server:app --host 0.0.0.0 --port 8000

This is how the hosted demo runs (Dockerfile.vercel). Locally, `make veroniqa` is simpler. The
pages themselves are in site.py.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from starlette.routing import Route

from agentqa.veroniqa import site

APP = Path(__file__).with_name("app.py")
INTRO = site.INTRO  # captions are burned in
POSTER = INTRO.with_suffix(".jpg")
TALK = "talk"  # Streamlit's base path: the app, its assets and its WebSocket live under /talk/

# Must be set before st.App builds its routes. Plain `streamlit run` (make veroniqa) is unaffected.
st.config.set_option("server.baseUrlPath", TALK)

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; media-src 'self'; style-src 'unsafe-inline'; img-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Cache-Control": "public, max-age=300",
}
CACHE = {"Cache-Control": "public, max-age=86400"}


async def home(request: Request) -> Response:
    return HTMLResponse(site.home_page(), headers=SECURITY_HEADERS)


async def robots(request: Request) -> Response:
    return PlainTextResponse(site.robots_txt(), headers=CACHE)


async def sitemap(request: Request) -> Response:
    return Response(site.sitemap_xml(), media_type="application/xml", headers=CACHE)


async def llms(request: Request) -> Response:
    return PlainTextResponse(site.llms_txt(), headers=CACHE)


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


app = st.App(
    APP,
    routes=[
        Route("/", home),
        Route("/watch", to_home),  # the home page's earlier address
        Route(f"/{TALK}", to_talk),
        Route("/robots.txt", robots),
        Route("/sitemap.xml", sitemap),
        Route("/llms.txt", llms),
        Route("/favicon.svg", favicon),
        Route("/videos/veroniqa-intro.mp4", intro),
        Route("/videos/veroniqa-intro.jpg", poster),
    ],
)
