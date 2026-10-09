"""VeroniQA as one ASGI app: the Streamlit UI at /, plus a watch page and the introduction video.

    uvicorn agentqa.veroniqa.server:app --host 0.0.0.0 --port 8000

This is how the hosted demo runs (Dockerfile.vercel). Locally, `make veroniqa` is simpler.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, Response
from starlette.routing import Route

ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).with_name("app.py")
INTRO = ROOT / "media/videos/veroniqa-intro.mp4"  # captions are burned in
POSTER = INTRO.with_suffix(".jpg")

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; media-src 'self'; style-src 'unsafe-inline'; img-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
CACHE = {"Cache-Control": "public, max-age=86400"}

WATCH_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meet VeroniQA</title>
<meta name="description" content="VeroniQA, the AI assistant for API testing built on AgentQA.">
<style>
  :root { --bg:#0b1020; --fg:#e8ecf6; --muted:#9aa3b8; --accent:#5b8cff; }
  @media (prefers-color-scheme: light) {
    :root { --bg:#f6f7fb; --fg:#141824; --muted:#556; --accent:#2f5fe0; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  main { max-width:1080px; margin:0 auto; padding:32px 16px 64px; }
  h1 { font-size:clamp(28px,5vw,44px); margin:0 0 8px; letter-spacing:-.02em; }
  p.lead { color:var(--muted); margin:0 0 24px; max-width:60ch; }
  video { width:100%; aspect-ratio:16/9; border-radius:12px; background:#000; display:block; }
  .cta { display:inline-block; margin:24px 0 0; padding:12px 20px; border-radius:10px;
         background:var(--accent); color:#fff; text-decoration:none; font-weight:600; }
  footer { color:var(--muted); font-size:14px; margin-top:40px; }
</style>
</head>
<body>
<main>
  <h1>Meet VeroniQA</h1>
  <p class="lead">The AI assistant for API testing, built on AgentQA. Give her your requirements,
  documents and links; she answers with sources, runs the tests and explains what she found.</p>
  <video src="/videos/veroniqa-intro.mp4" poster="/videos/veroniqa-intro.jpg" controls
    preload="metadata" playsinline></video>
  <a class="cta" href="/">Talk to VeroniQA</a>
  <footer>A proof of concept. The numbers in the video come from measured runs; the public demo
  uses simulated models.</footer>
</main>
</body>
</html>"""


async def watch(request: Request) -> Response:
    return HTMLResponse(WATCH_PAGE, headers=SECURITY_HEADERS)


async def intro(request: Request) -> Response:
    return FileResponse(INTRO, media_type="video/mp4", headers=CACHE)


async def poster(request: Request) -> Response:
    return FileResponse(POSTER, media_type="image/jpeg", headers=CACHE)


app = st.App(
    APP,
    routes=[
        Route("/watch", watch),
        Route("/videos/veroniqa-intro.mp4", intro),
        Route("/videos/veroniqa-intro.jpg", poster),
    ],
)
