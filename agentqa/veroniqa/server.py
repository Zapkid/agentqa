"""VeroniQA as one ASGI app: the Streamlit UI at /, plus a watch page and the videos.

    uvicorn agentqa.veroniqa.server:app --host 0.0.0.0 --port 8000

This is how the hosted demo runs (Dockerfile.vercel). Locally, `make veroniqa` is simpler.
"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).with_name("app.py")
VIDEOS = ROOT / "media/videos"
INTRO_SRT = VIDEOS / "veroniqa-intro.srt"

PROMOS = [
    ("Cost versus quality", "01-cost-vs-quality.mp4"),
    ("Trust, measured", "02-trust-measured.mp4"),
    ("Meet VeroniQA in the app", "03-meet-veroniqa.mp4"),
]
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; media-src 'self'; style-src 'unsafe-inline'; img-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def srt_to_vtt(srt: str) -> str:
    """WebVTT for the <track> element: a header, and '.' instead of ',' in timestamps."""
    body = re.sub(r"(\d\d:\d\d:\d\d),(\d\d\d)", r"\1.\2", srt.replace("\r\n", "\n"))
    return "WEBVTT\n\n" + body.strip() + "\n"


def watch_page() -> str:
    promos = "\n".join(
        f'<figure><video src="/videos/{name}" poster="/videos/{name[:-4]}.jpg" controls '
        f'preload="metadata" playsinline></video>'
        f"<figcaption>{title}</figcaption></figure>"
        for title, name in PROMOS
        if (VIDEOS / name).exists()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meet VeroniQA</title>
<meta name="description" content="VeroniQA, the AI assistant for API testing built on AgentQA.">
<style>
  :root {{ --bg:#0b1020; --fg:#e8ecf6; --muted:#9aa3b8; --accent:#5b8cff; --card:#141b30; }}
  @media (prefers-color-scheme: light) {{
    :root {{ --bg:#f6f7fb; --fg:#141824; --muted:#556; --accent:#2f5fe0; --card:#fff; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
  main {{ max-width:1080px; margin:0 auto; padding:32px 16px 64px; }}
  h1 {{ font-size:clamp(28px,5vw,44px); margin:0 0 8px; letter-spacing:-.02em; }}
  p.lead {{ color:var(--muted); margin:0 0 24px; max-width:60ch; }}
  video {{ width:100%; aspect-ratio:16/9; border-radius:12px; background:#000; display:block; }}
  .cta {{ display:inline-block; margin:20px 0 40px; padding:12px 20px; border-radius:10px;
         background:var(--accent); color:#fff; text-decoration:none; font-weight:600; }}
  h2 {{ font-size:20px; margin:0 0 16px; }}
  .grid {{ display:grid; gap:20px; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); }}
  figure {{ margin:0; background:var(--card); border-radius:14px; padding:10px; }}
  figcaption {{ padding:10px 4px 2px; font-weight:600; }}
  footer {{ color:var(--muted); font-size:14px; margin-top:40px; }}
</style>
</head>
<body>
<main>
  <h1>Meet VeroniQA</h1>
  <p class="lead">The AI assistant for API testing, built on AgentQA. Give her your requirements,
  documents and links; she answers with sources, runs the tests and explains what she found.</p>
  <video src="/videos/veroniqa-intro.mp4" poster="/videos/veroniqa-intro.jpg" controls
    preload="metadata" playsinline>
    <track kind="captions" src="/videos/veroniqa-intro.vtt" srclang="en" label="English" default>
  </video>
  <a class="cta" href="/">Talk to VeroniQA</a>
  <h2>Short films</h2>
  <div class="grid">{promos}</div>
  <footer>A proof of concept. Every number in the films comes from measured runs; the public demo
  uses simulated models.</footer>
</main>
</body>
</html>"""


async def watch(request: Request) -> Response:
    return HTMLResponse(watch_page(), headers=SECURITY_HEADERS)


async def captions(request: Request) -> Response:
    vtt = srt_to_vtt(INTRO_SRT.read_text(encoding="utf-8"))
    return PlainTextResponse(vtt, media_type="text/vtt", headers={"Cache-Control": "max-age=3600"})


app = st.App(
    APP,
    routes=[
        Route("/watch", watch),
        Route("/videos/veroniqa-intro.vtt", captions),
        Mount("/videos", StaticFiles(directory=VIDEOS), name="videos"),
    ],
)
