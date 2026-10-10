"""The public site around the VeroniQA app: the home page, and the files search engines and AI
assistants read (robots.txt, sitemap.xml, llms.txt).

Everything here is static text built once from data in the repository (the video's subtitles and
its length), so the pages make no claim the repo cannot back. VERONIQA_SITE_URL sets the absolute
address used in canonical links, the sitemap and structured data.
"""

from __future__ import annotations

import html
import json
import os
import re
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTRO = ROOT / "media/videos/veroniqa-intro.mp4"
SUBTITLES = INTRO.with_suffix(".srt")
REPO_URL = "https://github.com/Zapkid/agentqa"
AUTHOR = "Rowan Kendal"
AUTHOR_URL = "https://www.rowan-kendal.com/"
CONTACT_URL = f"{REPO_URL}/issues"
SECURITY_CONTACT = f"{REPO_URL}/security/advisories/new"
VIDEO_SECONDS = 85  # media/videos/veroniqa-intro.mp4 (85.2 s)
VIDEO_DATE = "2026-10-09"

NAME = "VeroniQA"
TITLE = "VeroniQA: the AI assistant for API testing"
DESCRIPTION = (
    "VeroniQA is an AI assistant for API testing, built on AgentQA. Give her your requirements, "
    "documents and links; she answers with cited sources, generates and runs API tests, and "
    "explains what she found."
)

FEATURES = [
    (
        "Answers with sources",
        "Ask about your API and get answers grounded in your own documents, with numbered "
        "citations. Citations that do not match a retrieved passage are dropped, and "
        '"not found" is a real answer.',
    ),
    (
        "Tests from your requirements",
        "Say “run the tests”. AgentQA plans risk-ranked tests from the spec and your "
        "documents, runs them, and triages each failure with evidence.",
    ),
    (
        "Knowledge you can trust",
        "Upload Markdown, text, HTML or PDF, or add links. Links are fetched from public "
        "addresses only, and text that looks like a prompt injection is quarantined.",
    ),
    (
        "Retrieval that fits the domain",
        "Each project has its own vector store (Chroma) combined with keyword search (BM25), so "
        "field names and error codes are found as reliably as plain prose.",
    ),
    (
        "Cost-aware by design",
        "A cheap model routes each message; stronger models are used only where they change the "
        "result. Every step is traced with its tokens and cost.",
    ),
    (
        "Fits how teams work",
        "Use the browser, let coding agents call AgentQA over MCP, or gate pull requests on an "
        "eval and a performance smoke test in CI.",
    ),
]

STEPS = [
    ("Create a project", "One folder per API, with its spec, settings and knowledge."),
    ("Add what your team already has", "Requirement documents, PDFs and links to your docs."),
    ("Ask, then run the tests", "Get cited answers, a test run, findings and a short report."),
]

FAQ = [
    (
        "What is VeroniQA?",
        "VeroniQA is a chat assistant for API testing. She keeps a project's requirements in a "
        "searchable knowledge base, answers questions with citations, and starts AgentQA test "
        "runs that plan, generate, execute and triage API tests.",
    ),
    (
        "What is AgentQA?",
        "AgentQA is the engine behind VeroniQA: an observable, guarded multi-agent "
        "system that turns an OpenAPI spec and requirement documents into risk-ranked tests, a "
        "run, triaged findings and an executive summary.",
    ),
    (
        "Does the online demo use a real AI model?",
        "No. The public demo uses deterministic simulated models, so it needs no API keys and "
        "costs nothing to try. Answers are extracts from the documents. Run it yourself with "
        "your own keys for real model answers.",
    ),
    (
        "Is what I upload private?",
        "Each visitor gets a private, temporary workspace tied to their link. Workspaces are "
        "deleted after 12 hours or when the server restarts. Do not upload confidential "
        "documents to the demo.",
    ),
    (
        "Can I test my own API?",
        "In the demo, tests run only against the bundled Orders API, which has planted bugs to "
        "find. To test your own API, run VeroniQA from the GitHub repository.",
    ),
    (
        "Is it production-ready?",
        "No. It is a proof of concept that shows the approach end to end. Fitting it to a "
        "specific team's APIs, models and controls comes next.",
    ),
]


def site_url() -> str:
    return os.environ.get("VERONIQA_SITE_URL", "https://veroniqa.vercel.app").rstrip("/")


def iso_duration(seconds: int) -> str:
    return f"PT{seconds // 60}M{seconds % 60}S" if seconds >= 60 else f"PT{seconds}S"


def transcript() -> list[str]:
    """The narration as paragraphs, from the subtitle file (index and timing lines dropped)."""
    if not SUBTITLES.exists():
        return []
    blocks = re.split(r"\n\s*\n", SUBTITLES.read_text(encoding="utf-8").strip())
    lines = []
    for block in blocks:
        text = [ln for ln in block.splitlines() if ln.strip() and "-->" not in ln]
        if text and text[0].strip().isdigit():
            text = text[1:]
        if text:
            lines.append(" ".join(t.strip() for t in text))
    return lines


def structured_data() -> dict[str, object]:
    url = site_url()
    return {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "WebSite",
                "@id": f"{url}/#website",
                "url": f"{url}/",
                "name": NAME,
                "description": DESCRIPTION,
                "inLanguage": "en",
                "publisher": {"@id": f"{url}/#author"},
            },
            {
                "@type": "Person",
                "@id": f"{url}/#author",
                "name": AUTHOR,
                "url": AUTHOR_URL,
            },
            {
                "@type": "SoftwareApplication",
                "@id": f"{url}/#app",
                "name": NAME,
                "url": f"{url}/talk/",
                "applicationCategory": "DeveloperApplication",
                "applicationSubCategory": "API testing",
                "operatingSystem": "Web",
                "description": DESCRIPTION,
                "isBasedOn": REPO_URL,
                "codeRepository": REPO_URL,
                "programmingLanguage": "Python",
                "featureList": [title for title, _ in FEATURES],
            },
            {
                "@type": "VideoObject",
                "@id": f"{url}/#video",
                "name": "Meet VeroniQA",
                "description": DESCRIPTION,
                "thumbnailUrl": f"{url}/videos/veroniqa-intro.jpg",
                "contentUrl": f"{url}/videos/veroniqa-intro.mp4",
                "uploadDate": VIDEO_DATE,
                "duration": iso_duration(VIDEO_SECONDS),
                "inLanguage": "en",
                "transcript": " ".join(transcript()),
            },
            {
                "@type": "FAQPage",
                "@id": f"{url}/#faq",
                "mainEntity": [
                    {
                        "@type": "Question",
                        "name": q,
                        "acceptedAnswer": {"@type": "Answer", "text": a},
                    }
                    for q, a in FAQ
                ],
            },
        ],
    }


LOGO = (
    '<svg viewBox="0 0 64 64" width="{size}" height="{size}" aria-hidden="true" focusable="false">'
    '<defs><linearGradient id="vq-g{size}" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#a99bff"/><stop offset="1" stop-color="#7c6dfa"/>'
    "</linearGradient></defs>"
    '<rect x="4" y="4" width="56" height="46" rx="14" fill="url(#vq-g{size})"/>'
    '<path d="M18 50 L14 60 L28 50 Z" fill="#7c6dfa"/>'
    '<path d="M20 27 L29 36 L45 18" fill="none" stroke="#fff" stroke-width="6" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)

FAVICON = LOGO.replace('width="{size}" height="{size}" aria-hidden="true" focusable="false"', "")
FAVICON = FAVICON.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1).replace(
    "{size}", "f"
)

# Headline numbers for the hero tiles. They come from the latest results/*.json (the same files
# as the README's Results section); tests/test_veroniqa_hosted.py fails if they drift. The image
# does not ship results/, so the values are kept here.
STATS = {
    "recall_pct": 94,  # S3 (cost-aware routing): mean share of planted bugs found
    "cost_pct": 47,  # S3 cost as a share of S0 (strongest model for everything)
    "perf_caught": 6,  # performance defects flagged
    "perf_total": 6,
    "planted_bugs": 12,  # functional bugs planted in the bundled Orders API
    "cost_levers": 10,  # mechanisms switched off one at a time (S3-no-* ablations)
}

_ICON = {
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/>'
    '<circle cx="12" cy="12" r="1"/>',
    "coins": '<circle cx="9" cy="9" r="6"/><path d="M18.1 10.4a6 6 0 1 1-7.7 7.7"/>'
    '<path d="M8 7h2v4"/>',
    "gauge": '<path d="M12 14l4-4"/><path d="M3.3 19a10 10 0 1 1 17.4 0"/>',
    "bug": '<rect x="8" y="6" width="8" height="14" rx="4"/><path d="M19 7l-3 2M5 7l3 2'
    'M19 19l-3-2M5 19l3-2M20 13h-4M4 13h4M10 4l1 2M14 4l-1 2"/>',
    "sliders": '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
    "key": '<circle cx="7.5" cy="15.5" r="5.5"/><path d="M21 2l-9.6 9.6M15.5 7.5l3 3L22 7l-3-3"/>',
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "chat": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "github": '<path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.9a3.4 3.4 0 0 0-.9-2.6c3.1-.3 6.4-1.5 '
    "6.4-7A5.4 5.4 0 0 0 20 4.8 5 5 0 0 0 19.9 1S18.7.7 16 2.5a13.4 13.4 0 0 0-7 0C6.3.7 "
    "5.1 1 5.1 1A5 5 0 0 0 5 4.8a5.4 5.4 0 0 0-1.5 3.7c0 5.4 3.3 6.6 6.4 7A3.4 3.4 0 0 0 9 "
    '18.1V22"/>',
}


def _icon(name: str, size: int = 20) -> str:
    return (
        f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
        'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true" focusable="false">{_ICON[name]}</svg>'
    )


def stat_tiles() -> list[tuple[str, str, str]]:
    """(icon, value, caption) for the hero grid."""
    s = STATS
    return [
        ("target", f"{s['recall_pct']}%", "of planted bugs found"),
        ("coins", f"{s['cost_pct']}%", "of the cost of the strongest model"),
        ("gauge", f"{s['perf_caught']}/{s['perf_total']}", "performance defects caught"),
        ("bug", str(s["planted_bugs"]), "bugs planted in the demo API"),
        ("sliders", str(s["cost_levers"]), "cost levers, each measured"),
        ("key", "0", "API keys needed to try it"),
    ]


STACK = [
    "Python 3.12",
    "FastAPI",
    "Streamlit",
    "Chroma vector store",
    "BM25 hybrid retrieval",
    "OpenAPI 3 · Swagger 2",
    "pytest",
    "OpenTelemetry",
    "Model Context Protocol",
    "Locust",
    "Claude",
    "Gemini",
    "OpenRouter",
    "Prompt-injection quarantine",
    "SSRF-safe links",
    "Replay-gated CI",
]

STYLE = """
:root { color-scheme:dark; --bg:#080c1a; --surface:#111118; --surface-2:#18181f; --line:#2a2a35;
        --fg:#e8e8f0; --soft:#c8c8d8; --muted:#8888a0; --accent:#7c6dfa; --accent-2:#a99bff;
        --teal:#5eead4; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; background:var(--bg); overflow-x:clip; }
body { margin:0; color:var(--fg); background:var(--bg); -webkit-font-smoothing:antialiased;
       font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif; }
::selection { color:var(--accent-2); background:#7c6dfa33; }
a { color:var(--accent-2); }
.wrap { max-width:1100px; margin:0 auto; padding:0 16px; }
.skip { position:absolute; left:-999px; }
.skip:focus { left:16px; top:8px; background:var(--surface); padding:8px 12px; z-index:10; }
.grad { background:linear-gradient(90deg,var(--accent-2),var(--teal)); -webkit-background-clip:text;
        background-clip:text; color:transparent; }

header.site { position:sticky; top:0; z-index:5; background:#0a0a0fcc; backdrop-filter:blur(12px);
              border-bottom:1px solid var(--line); }
header.site .wrap { display:flex; align-items:center; gap:20px; min-height:60px; flex-wrap:wrap; }
.brand { display:flex; align-items:center; gap:10px; font-weight:700; font-size:17px;
         color:var(--fg); text-decoration:none; }
nav.primary { margin-left:auto; display:flex; gap:22px; align-items:center; flex-wrap:wrap; }
.nav-link { font-size:13px; text-decoration:none; white-space:nowrap; color:transparent;
            background:linear-gradient(90deg,var(--accent-2) 0%,var(--teal) 49%,var(--muted) 51% 100%)
            100% 0/200% 100%; -webkit-background-clip:text; background-clip:text;
            transition:background-position .4s; }
.nav-link:hover, .nav-link:focus-visible { background-position:0 0; }
.pill-btn { color:var(--accent-2); background:#7c6dfa24; border:1px solid #7c6dfa73;
            border-radius:100px; padding:7px 16px; font-size:13px; font-weight:600;
            text-decoration:none; white-space:nowrap; transition:background-color .2s,color .2s; }
.pill-btn:hover, .pill-btn:focus-visible { color:#fff; background:#7c6dfa47; border-color:#7c6dfacc; }

.btn { display:inline-flex; align-items:center; gap:8px; padding:12px 22px; border-radius:8px;
       font-size:14px; font-weight:600; text-decoration:none; color:var(--fg);
       border:1px solid var(--line); transition:border-color .2s,background-color .2s; }
.btn:hover, .btn:focus-visible { border-color:var(--accent); background:#7c6dfa0f; }
.btn.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
.btn.primary:hover, .btn.primary:focus-visible { background:#9187fb; }

.hero { position:relative; padding:72px 0 64px; overflow:hidden; }
.hero-glow { position:absolute; top:-80px; left:-120px; width:600px; height:600px; border-radius:50%;
             background:radial-gradient(circle,#7c6dfa2e,transparent 70%); filter:blur(80px);
             pointer-events:none; animation:drift 8s ease-in-out infinite alternate; }
@keyframes drift { from { transform:translate(-10%,-10%) scale(1); } to { transform:translate(10%,10%) scale(1.1); } }
.hero-grid { position:relative; z-index:1; display:grid; gap:56px; align-items:center;
             grid-template-columns:minmax(0,1fr) 400px; }
.status { display:inline-flex; align-items:center; gap:8px; padding:5px 14px; border-radius:100px;
          border:1px solid #7c6dfa59; background:#7c6dfa14; color:var(--accent-2); font-size:13px;
          margin:0 0 22px; }
.status::before { content:""; width:8px; height:8px; border-radius:50%; background:#22c55e;
                  box-shadow:0 0 10px #22c55e; }
h1 { font-size:clamp(44px,7vw,72px); line-height:1.02; letter-spacing:-.035em; margin:0 0 14px; }
.tagline { font-size:clamp(19px,2.4vw,23px); font-weight:500; margin:0 0 16px; }
.lead { color:var(--teal); font-size:16.5px; max-width:58ch; margin:0 0 30px; opacity:.92; }
.actions { display:flex; gap:12px; flex-wrap:wrap; }
.tiles { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:0; padding:0; list-style:none; }
.tile { background:#111118eb; border:1px solid #2a2a35e6; border-radius:12px; padding:16px 10px;
        min-height:126px; display:flex; flex-direction:column; align-items:center; justify-content:center;
        gap:6px; text-align:center; box-shadow:inset 0 1px #ffffff0a; transition:border-color .2s; }
.tile:hover { border-color:#7c6dfa66; }
.tile svg { color:#7c6dfa; }
.tile b { font-size:30px; line-height:1.1; font-weight:700; letter-spacing:-.02em; }
.tile span { color:var(--muted); font-size:12px; line-height:1.35; }
.tiles-note { color:var(--muted); font-size:12px; margin:10px 2px 0; text-align:right; }

.marquee { overflow:hidden; padding:6px 0 0;
           -webkit-mask-image:linear-gradient(90deg,transparent,#000 8% 92%,transparent);
           mask-image:linear-gradient(90deg,transparent,#000 8% 92%,transparent); }
.marquee-track { display:flex; gap:10px; width:max-content; margin:0; padding:0; list-style:none;
                 animation:marquee 40s linear infinite; }
.marquee:hover .marquee-track { animation-play-state:paused; }
@keyframes marquee { to { transform:translateX(-50%); } }
.marquee li { color:var(--muted); background:var(--surface); border:1px solid var(--line);
              border-radius:100px; padding:5px 14px; font-size:13px; white-space:nowrap; }

section { padding:88px 0; border-top:1px solid var(--line); }
.eyebrow { display:flex; align-items:center; gap:8px; color:var(--accent-2); font-size:12.5px;
           font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin:0 0 12px; }
.eyebrow::before { content:""; width:6px; height:6px; border-radius:50%; background:var(--accent); }
h2 { font-size:clamp(28px,4vw,40px); line-height:1.15; letter-spacing:-.025em; margin:0 0 12px; }
.section-lead { color:var(--muted); font-size:16.5px; margin:0 0 36px; max-width:62ch; }

.frame { padding:2px; border-radius:16px; background:linear-gradient(var(--surface),var(--surface)) padding-box,
         conic-gradient(from var(--angle),#6366f1,#7c6dfa,#5eead4,#a99bff,#6366f1) border-box;
         border:2px solid transparent; animation:spin 4s linear infinite; }
@property --angle { syntax:"<angle>"; inherits:false; initial-value:0deg; }
@keyframes spin { to { --angle:360deg; } }
video { width:100%; aspect-ratio:16/9; border-radius:13px; background:#000; display:block; }
figure.video { margin:0; }
figcaption { color:var(--muted); font-size:14px; margin-top:14px; }

.grid { display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); }
.card { background:var(--surface-2); border:1px solid var(--line); border-radius:10px; padding:24px;
        transition:border-color .2s,transform .2s; }
.card:hover { border-color:#7c6dfa66; transform:translateY(-2px); }
.card h3 { margin:0 0 8px; font-size:17px; }
.card p { margin:0; color:var(--muted); font-size:15px; }
.card .num { display:inline-block; font-size:12px; font-weight:700; letter-spacing:.06em;
             color:var(--accent-2); margin-bottom:10px; }

ol.steps { list-style:none; padding:0; margin:0; display:grid; gap:16px;
           grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); counter-reset:step; }
ol.steps li { counter-increment:step; background:var(--surface); border:1px solid var(--line);
              border-radius:12px; padding:22px; }
ol.steps li::before { content:"0" counter(step); display:block; font-size:28px; font-weight:700;
              letter-spacing:-.02em; margin-bottom:8px; background:linear-gradient(90deg,var(--accent-2),var(--teal));
              -webkit-background-clip:text; background-clip:text; color:transparent; }
ol.steps h3 { margin:0 0 4px; font-size:17px; }
ol.steps p { margin:0; color:var(--muted); font-size:15px; }
.cta-row { margin-top:32px; }

.faq details { max-width:820px; }
details { background:var(--surface); border:1px solid var(--line); border-radius:12px;
          padding:16px 20px; margin-bottom:10px; transition:border-color .2s; }
details[open], details:hover { border-color:#7c6dfa66; }
summary { cursor:pointer; font-weight:600; list-style:none; display:flex; justify-content:space-between; gap:16px; }
summary::-webkit-details-marker { display:none; }
summary::after { content:"+"; color:var(--accent-2); font-weight:400; font-size:20px; line-height:1; }
details[open] summary::after { content:"\\2212"; }
details p { color:var(--muted); margin:10px 0 0; }
.transcript { color:var(--soft); }
.transcript p { margin:0 0 10px; max-width:75ch; }

.closing { text-align:center; }
.closing .section-lead { margin-left:auto; margin-right:auto; }
.closing .actions { justify-content:center; }

footer.site { border-top:1px solid var(--line); padding:36px 0 52px; color:var(--muted); font-size:14px; }
footer.site .wrap { display:flex; gap:24px; flex-wrap:wrap; justify-content:space-between; align-items:flex-start; }
footer.site p { margin:10px 0 0; max-width:46ch; }
footer.site nav { display:flex; gap:18px; flex-wrap:wrap; }
footer.site a { color:var(--muted); text-decoration:none; }
footer.site a:hover { color:var(--accent-2); }

@media (max-width:1000px) { .hero-grid { grid-template-columns:1fr; gap:44px; } .tiles-note { text-align:left; } }
@media (max-width:640px) {
  nav.primary .nav-link { display:none; }
  .hero { padding:44px 0 48px; }
  .hero-glow { width:280px; height:280px; top:-40px; left:-40px; }
  .tiles { gap:8px; } .tile { min-height:104px; padding:12px 6px; } .tile b { font-size:24px; }
  section { padding:60px 0; }
}
@media (prefers-reduced-motion:reduce) {
  html { scroll-behavior:auto; }
  *, ::before, ::after { animation:none !important; transition:none !important; }
}
"""


def _e(text: str) -> str:
    return html.escape(text, quote=True)


def _header() -> str:
    return f"""<header class="site">
  <div class="wrap">
    <a class="brand" href="/" aria-label="VeroniQA home">{LOGO.format(size=28)}<span>VeroniQA</span></a>
    <nav class="primary" aria-label="Main">
      <a class="nav-link" href="/#watch">Watch</a>
      <a class="nav-link" href="/#features">Features</a>
      <a class="nav-link" href="/#how-it-works">How it works</a>
      <a class="nav-link" href="/#faq">FAQ</a>
      <a class="nav-link" href="{REPO_URL}" rel="noopener">GitHub</a>
      <a class="pill-btn" href="/talk/">Talk to VeroniQA</a>
    </nav>
  </div>
</header>"""


def _footer() -> str:
    return f"""<footer class="site">
  <div class="wrap">
    <div>
      <a class="brand" href="/" aria-label="VeroniQA home">{LOGO.format(size=22)}<span>VeroniQA</span></a>
      <p>A proof of concept built on AgentQA. The public demo uses simulated models. Built by
      <a href="https://www.rowan-kendal.com/" rel="noopener">Rowan Kendal</a>.</p>
    </div>
    <nav aria-label="Footer">
      <a href="/">Home</a>
      <a href="/talk/">Talk to VeroniQA</a>
      <a href="{REPO_URL}" rel="noopener">Source code</a>
      <a href="/privacy">Privacy</a>
      <a href="/.well-known/security.txt">Security</a>
      <a href="{CONTACT_URL}" rel="noopener">Contact</a>
      <a href="/index.md">Markdown</a>
      <a href="/llms.txt">llms.txt</a>
      <a href="/sitemap.xml">Sitemap</a>
    </nav>
  </div>
</footer>"""


def _shell(title: str, description: str, path: str, body: str, *, index: bool = True) -> str:
    """A simple page (privacy, 404) with the site's header, footer and styles."""
    url = site_url()
    robots = "index, follow" if index else "noindex"
    md = f"{path}.md" if path not in ("", "/") else "/index.md"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<meta name="description" content="{_e(description)}">
<meta name="robots" content="{robots}">
<link rel="canonical" href="{url}{path}">
<link rel="alternate" type="text/markdown" href="{md}">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<meta name="theme-color" content="#080c1a">
<meta name="color-scheme" content="dark">
<meta name="author" content="{AUTHOR}">
<style>{STYLE}
.doc {{ padding-top:64px; padding-bottom:88px; }} .doc h1 {{ font-size:clamp(34px,5vw,48px); }}
.doc h2 {{ font-size:22px; margin:36px 0 10px; }} .doc p, .doc li {{ color:var(--soft); max-width:72ch; }}
.doc ul {{ padding-left:20px; }}</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
{_header()}
<main id="main"><div class="wrap doc">
{body}
</div></main>
{_footer()}
</body>
</html>"""


@cache
def home_page() -> str:
    url = site_url()
    tiles = "\n".join(
        f'<li class="tile">{_icon(icon)}<b class="grad">{_e(value)}</b><span>{_e(caption)}</span></li>'
        for icon, value, caption in stat_tiles()
    )
    stack = "".join(f"<li>{_e(item)}</li>" for item in STACK)
    features = "\n".join(
        f'<article class="card"><span class="num">{i:02d}</span><h3>{_e(t)}</h3><p>{_e(d)}</p></article>'
        for i, (t, d) in enumerate(FEATURES, 1)
    )
    steps = "\n".join(f"<li><h3>{_e(t)}</h3><p>{_e(d)}</p></li>" for t, d in STEPS)
    faq = "\n".join(f"<details><summary>{_e(q)}</summary><p>{_e(a)}</p></details>" for q, a in FAQ)
    words = "\n".join(f"<p>{_e(line)}</p>" for line in transcript())
    ld = json.dumps(structured_data(), ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(TITLE)}</title>
<meta name="description" content="{_e(DESCRIPTION)}">
<meta name="robots" content="index, follow, max-image-preview:large, max-video-preview:-1">
<link rel="canonical" href="{url}/">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<meta name="theme-color" content="#080c1a">
<meta name="color-scheme" content="dark">
<meta property="og:type" content="website">
<meta property="og:site_name" content="{NAME}">
<meta property="og:title" content="{_e(TITLE)}">
<meta property="og:description" content="{_e(DESCRIPTION)}">
<meta property="og:url" content="{url}/">
<meta property="og:image" content="{url}/videos/veroniqa-intro.jpg">
<meta property="og:image:width" content="1280">
<meta property="og:image:height" content="720">
<meta property="og:image:alt" content="VeroniQA, your AI assistant for API testing">
<meta property="og:video" content="{url}/videos/veroniqa-intro.mp4">
<meta property="og:video:type" content="video/mp4">
<meta property="og:video:width" content="1920">
<meta property="og:video:height" content="1080">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{_e(TITLE)}">
<meta name="twitter:description" content="{_e(DESCRIPTION)}">
<meta name="twitter:image" content="{url}/videos/veroniqa-intro.jpg">
<link rel="alternate" type="text/markdown" href="/index.md" title="This page as Markdown">
<link rel="alternate" type="text/plain" href="/llms.txt" title="VeroniQA for language models">
<link rel="manifest" href="/site.webmanifest">
<meta name="author" content="{AUTHOR}">
<meta name="application-name" content="{NAME}">
<meta property="og:locale" content="en_US">
<script type="application/ld+json">{ld}</script>
<style>{STYLE}</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
{_header()}
<main id="main">
  <div class="hero">
    <div class="hero-glow" aria-hidden="true"></div>
    <div class="wrap hero-grid">
      <div>
        <p class="status">Live demo · no sign-up, no API keys</p>
        <h1>VeroniQA</h1>
        <p class="tagline grad">The AI assistant for API testing, built on AgentQA</p>
        <p class="lead">Give her your requirements, documents and API docs. She answers with sources,
        writes and runs the tests, and explains every failure with evidence.</p>
        <div class="actions">
          <a class="btn primary" href="/talk/">{_icon("chat", 18)} Talk to VeroniQA</a>
          <a class="btn" href="#watch">{_icon("play", 16)} Watch the intro</a>
        </div>
      </div>
      <div>
        <ul class="tiles" aria-label="Measured results">
{tiles}
        </ul>
        <p class="tiles-note">Measured on the simulated benchmark in the repository.</p>
      </div>
    </div>
  </div>
  <div class="wrap">
    <div class="marquee" aria-label="Built with">
      <ul class="marquee-track">{stack}<li aria-hidden="true">{'</li><li aria-hidden="true">'.join(_e(s) for s in STACK)}</li></ul>
    </div>
  </div>

  <section id="watch" aria-labelledby="watch-title">
    <div class="wrap">
      <p class="eyebrow">Watch</p>
      <h2 id="watch-title">Meet VeroniQA <span class="grad">in 85 seconds</span></h2>
      <p class="section-lead">Knowledge in, cited answers out, safe links, and a full test run from
      one chat message.</p>
      <figure class="video">
        <div class="frame">
          <video src="/videos/veroniqa-intro.mp4" poster="/videos/veroniqa-intro.jpg" controls
            preload="metadata" playsinline aria-label="Meet VeroniQA, an 85-second introduction"></video>
        </div>
        <figcaption>Captions are built in; the <a href="#transcript">transcript</a> is below.</figcaption>
      </figure>
    </div>
  </section>

  <section id="features" aria-labelledby="features-title">
    <div class="wrap">
      <p class="eyebrow">Features</p>
      <h2 id="features-title">API testing, <span class="grad">AI-augmented</span></h2>
      <p class="section-lead">One place for an API's requirements, questions and test runs, on top of
      an engine that is observable, guarded and measured.</p>
      <div class="grid">
{features}
      </div>
    </div>
  </section>

  <section id="how-it-works" aria-labelledby="how-title">
    <div class="wrap">
      <p class="eyebrow">How it works</p>
      <h2 id="how-title">From documents to findings <span class="grad">in three steps</span></h2>
      <p class="section-lead">The demo starts you with a sample Orders API that has planted bugs, so
      you can try every step right away.</p>
      <ol class="steps">
{steps}
      </ol>
      <p class="cta-row"><a class="btn primary" href="/talk/">{_icon("chat", 18)} Try it now</a></p>
    </div>
  </section>

  <section id="faq" aria-labelledby="faq-title">
    <div class="wrap faq">
      <p class="eyebrow">FAQ</p>
      <h2 id="faq-title">Questions</h2>
{faq}
    </div>
  </section>

  <section id="transcript" aria-labelledby="transcript-title">
    <div class="wrap transcript">
      <p class="eyebrow">Transcript</p>
      <h2 id="transcript-title">Video transcript</h2>
{words}
    </div>
  </section>

  <section class="closing" aria-labelledby="closing-title">
    <div class="wrap">
      <h2 id="closing-title">Ready to <span class="grad">try it?</span></h2>
      <p class="section-lead">Your own private workspace, preloaded with the demo API. Nothing to
      install and nothing to pay.</p>
      <div class="actions">
        <a class="btn primary" href="/talk/">{_icon("chat", 18)} Talk to VeroniQA</a>
        <a class="btn" href="{REPO_URL}" rel="noopener">{_icon("github", 18)} Source on GitHub</a>
      </div>
    </div>
  </section>
</main>
{_footer()}
</body>
</html>"""


# Content Signals (contentsignals.org): this page may be indexed for search, used to answer
# questions (ai-input) and used for training (ai-train), in line with the crawlers allowed below.
CONTENT_SIGNAL = "Content-Signal: search=yes, ai-input=yes, ai-train=yes"


def robots_txt() -> str:
    url = site_url()
    lines = ["# Search engines and AI assistants are welcome.", ""]
    for agent in (
        "*",
        "GPTBot",
        "OAI-SearchBot",
        "ChatGPT-User",
        "ClaudeBot",
        "Claude-User",
        "Claude-SearchBot",
        "PerplexityBot",
        "Google-Extended",
        "Applebot-Extended",
    ):
        lines += [f"User-agent: {agent}", "Allow: /", "Disallow: /talk/_stcore/"]
        if agent == "*":
            lines.append(CONTENT_SIGNAL)
        lines.append("")
    lines.append(f"Sitemap: {url}/sitemap.xml")
    return "\n".join(lines) + "\n"


def sitemap_xml() -> str:
    url = site_url()
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:video="http://www.google.com/schemas/sitemap-video/1.1">
  <url>
    <loc>{url}/</loc>
    <lastmod>{VIDEO_DATE}</lastmod>
    <video:video>
      <video:thumbnail_loc>{url}/videos/veroniqa-intro.jpg</video:thumbnail_loc>
      <video:title>Meet VeroniQA</video:title>
      <video:description>{_e(DESCRIPTION)}</video:description>
      <video:content_loc>{url}/videos/veroniqa-intro.mp4</video:content_loc>
      <video:duration>{VIDEO_SECONDS}</video:duration>
      <video:publication_date>{VIDEO_DATE}</video:publication_date>
    </video:video>
  </url>
  <url>
    <loc>{url}/talk/</loc>
    <lastmod>{VIDEO_DATE}</lastmod>
  </url>
  <url>
    <loc>{url}/privacy</loc>
    <lastmod>{PRIVACY_DATE}</lastmod>
  </url>
</urlset>
"""


def llms_txt() -> str:
    """A plain summary for language models (the llms.txt convention)."""
    url = site_url()
    features = "\n".join(f"- {t}: {d}" for t, d in FEATURES)
    faq = "\n".join(f"- {q} {a}" for q, a in FAQ)
    return f"""# VeroniQA

> {DESCRIPTION}

VeroniQA is a proof of concept by {AUTHOR} ({AUTHOR_URL}). The public demo at {url}/ uses
simulated models and runs tests only against a bundled sample Orders API with planted bugs. No
account, sign-up or API key is needed: open {url}/talk/ and a private workspace is created.

## Links

- [Home page with the introduction video]({url}/index.md): Markdown; also served at {url}/ to
  clients that send `Accept: text/markdown`
- [Talk to VeroniQA (the app)]({url}/talk/): an interactive web app (it needs a browser with
  JavaScript); request it with `Accept: text/markdown` for a description
- [Privacy]({url}/privacy.md)
- [Security contact]({url}/.well-known/security.txt)
- [Source code: AgentQA and VeroniQA]({REPO_URL})
- [README]({REPO_URL}/blob/main/README.md)
- [Security notes]({REPO_URL}/blob/main/docs/SECURITY.md)
- [Known limitations]({REPO_URL}/blob/main/docs/LIMITATIONS.md)

## Features

{features}

## Frequently asked questions

{faq}

## Video transcript

{" ".join(transcript())}
"""


# ------------------------------------------------------------------ Markdown twins and trust pages

PRIVACY_DATE = "2026-10-10"

PRIVACY = [
    (
        "What this site collects",
        [
            "No accounts, no sign-up and no analytics or advertising trackers.",
            "The hosting provider (Vercel) keeps standard request logs (address, time, page) to run "
            "and protect the service.",
            "The app at /talk/ sets one technical cookie (`_streamlit_xsrf`) that protects its forms "
            "against cross-site request forgery. Nothing else is stored in your browser.",
        ],
    ),
    (
        "What you put into the demo",
        [
            "Projects, uploaded documents, fetched links and chat history live in a private, "
            "temporary workspace on the server, reachable only through your workspace link.",
            "Workspaces are deleted after 12 hours without use, and whenever the server restarts "
            "or is redeployed.",
            "The demo uses simulated models: nothing you upload or type is sent to an AI provider.",
            "When you add a link, the server fetches that page (only from public addresses) to "
            "index it.",
            "Do not upload confidential or personal data: this is a public demo.",
        ],
    ),
    (
        "Contact",
        [
            f"Questions or requests: open an issue at {CONTACT_URL}. Security reports: "
            f"{SECURITY_CONTACT} (see /.well-known/security.txt).",
        ],
    ),
]


def home_markdown() -> str:
    """The home page as Markdown, for agents that ask for text/markdown."""
    url = site_url()
    tiles = "\n".join(f"- **{v}** {c}" for _, v, c in stat_tiles())
    features = "\n".join(f"### {t}\n\n{d}\n" for t, d in FEATURES)
    steps = "\n".join(f"{i}. **{t}.** {d}" for i, (t, d) in enumerate(STEPS, 1))
    faq = "\n".join(f"### {q}\n\n{a}\n" for q, a in FAQ)
    words = "\n\n".join(transcript())
    return f"""# {TITLE}

> {DESCRIPTION}

- Try it: [{url}/talk/]({url}/talk/) (no sign-up, no API keys; a private workspace preloaded
  with a demo API is created for you)
- Watch: [Meet VeroniQA, 85 s video]({url}/videos/veroniqa-intro.mp4)
- Source code: [{REPO_URL}]({REPO_URL})
- Built by [{AUTHOR}]({AUTHOR_URL})

## Measured results

Measured on the simulated benchmark in the repository:

{tiles}

## Features

{features}
## How it works

{steps}

## Frequently asked questions

{faq}
## Video transcript

{words}

## More

- [Privacy]({url}/privacy.md)
- [Security contact]({url}/.well-known/security.txt)
- [llms.txt]({url}/llms.txt)
- [Sitemap]({url}/sitemap.xml)
"""


def talk_markdown() -> str:
    """What /talk/ is, for agents: the app itself needs a browser with JavaScript."""
    url = site_url()
    return f"""# Talk to VeroniQA

The VeroniQA app at {url}/talk/ is an interactive web app (Streamlit): it needs a browser with
JavaScript and keeps a live connection while you use it. There is no sign-up and no API key.

Opening it creates a private, temporary workspace (its id is kept in the page address as `?w=`)
with a demo project: a sample Orders API, its requirement documents and planted bugs. In the app
you can:

- **Chat:** ask about the API; answers cite the passages they come from.
- **Knowledge:** upload Markdown, text, HTML or PDF files, or add links, including OpenAPI or
  Swagger files and Swagger UI pages.
- **Test runs:** run AgentQA's tests against the demo API and read the findings.
- **New project:** create more projects in the same workspace.

The demo uses simulated models. To test your own API, run VeroniQA from {REPO_URL}.

- [Home page]({url}/index.md)
- [llms.txt]({url}/llms.txt)
"""


def privacy_markdown() -> str:
    parts = [
        f"# Privacy\n\nLast updated {PRIVACY_DATE}. This covers {site_url()}/ and the app at /talk/."
    ]
    for heading, items in PRIVACY:
        parts.append(f"## {heading}\n\n" + "\n".join(f"- {i}" for i in items))
    return "\n\n".join(parts) + "\n"


def _md_inline(text: str) -> str:
    """Escape, then render `code` spans and bare URLs."""
    out = _e(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return re.sub(r"(https://[^\s<]+[^\s<.,)])", r'<a href="\1" rel="noopener">\1</a>', out)


def privacy_page() -> str:
    body = [
        '<p class="eyebrow">Privacy</p>',
        "<h1>Privacy</h1>",
        f"<p>Last updated {PRIVACY_DATE}. This covers this site and the app at "
        '<a href="/talk/">/talk/</a>.</p>',
    ]
    for heading, items in PRIVACY:
        body.append(f"<h2>{_e(heading)}</h2><ul>")
        body += [f"<li>{_md_inline(i)}</li>" for i in items]
        body.append("</ul>")
    return _shell(
        "Privacy · VeroniQA",
        "What the VeroniQA demo stores, for how long, and how to reach the author.",
        "/privacy",
        "\n".join(body),
    )


def not_found_markdown(path: str) -> str:
    url = site_url()
    return f"""# Page not found (404)

There is nothing at `{path}`. Useful places:

- [Home page]({url}/index.md)
- [Talk to VeroniQA]({url}/talk/)
- [llms.txt]({url}/llms.txt): a summary of this site for language models
- [Sitemap]({url}/sitemap.xml)
"""


def not_found_page(path: str) -> str:
    body = f"""<p class="eyebrow">404</p>
<h1>Page not found</h1>
<p>There is nothing at <code>{_e(path)}</code>.</p>
<p class="actions" style="margin-top:28px"><a class="btn primary" href="/">Go to the home page</a>
<a class="btn" href="/talk/">Talk to VeroniQA</a></p>
<p>Agents: <a href="/llms.txt">llms.txt</a> summarises this site, and
<a href="/sitemap.xml">sitemap.xml</a> lists its pages.</p>"""
    return _shell("Page not found · VeroniQA", "This page does not exist.", path, body, index=False)


def security_txt(now: datetime | None = None) -> str:
    """RFC 9116. Expires is kept about six months ahead."""
    expires = (now or datetime.now(UTC)) + timedelta(days=180)
    url = site_url()
    return (
        f"Contact: {SECURITY_CONTACT}\n"
        f"Contact: {CONTACT_URL}\n"
        f"Expires: {expires.strftime('%Y-%m-%dT00:00:00Z')}\n"
        "Preferred-Languages: en\n"
        f"Canonical: {url}/.well-known/security.txt\n"
        f"Policy: {REPO_URL}/blob/main/docs/SECURITY.md\n"
    )


def web_manifest() -> str:
    return json.dumps(
        {
            "name": TITLE,
            "short_name": NAME,
            "description": DESCRIPTION,
            "start_url": "/",
            "display": "standalone",
            "background_color": "#080c1a",
            "theme_color": "#080c1a",
            "icons": [{"src": "/favicon.svg", "sizes": "any", "type": "image/svg+xml"}],
        },
        indent=2,
    )


def prefers_markdown(accept: str) -> bool:
    """True when the Accept header ranks text/markdown at least as high as text/html."""
    q: dict[str, float] = {}
    for part in accept.lower().split(","):
        media, *params = (x.strip() for x in part.split(";"))
        weight = 1.0
        for param in params:
            if param.startswith("q="):
                try:
                    weight = float(param[2:])
                except ValueError:
                    weight = 0.0
        q[media] = max(q.get(media, 0.0), weight)
    md = q.get("text/markdown", 0.0)
    html_q = max(q.get("text/html", 0.0), q.get("application/xhtml+xml", 0.0))
    return md > 0 and md >= html_q
