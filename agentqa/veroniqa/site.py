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
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTRO = ROOT / "media/videos/veroniqa-intro.mp4"
SUBTITLES = INTRO.with_suffix(".srt")
REPO_URL = "https://github.com/Zapkid/agentqa"
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
    '<rect x="4" y="4" width="56" height="46" rx="14" fill="#ee7b3a"/>'
    '<path d="M18 50 L14 60 L28 50 Z" fill="#ee7b3a"/>'
    '<path d="M20 27 L29 36 L45 18" fill="none" stroke="#fff" stroke-width="6" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)

FAVICON = LOGO.replace('width="{size}" height="{size}" aria-hidden="true" focusable="false"', "")
FAVICON = FAVICON.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)

STYLE = """
:root { --bg:#0b1020; --surface:#121a30; --line:#24304d; --fg:#e8ecf6; --muted:#a3acc2;
        --accent:#ee7b3a; --accent-ink:#1a0d04; --link:#8fb0ff; }
@media (prefers-color-scheme: light) {
  :root { --bg:#f7f8fb; --surface:#ffffff; --line:#e3e7ef; --fg:#141824; --muted:#5a6275;
          --accent:#d8611f; --accent-ink:#ffffff; --link:#2f5fe0; }
}
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
a { color:var(--link); }
.wrap { max-width:1120px; margin:0 auto; padding:0 16px; }
.skip { position:absolute; left:-999px; }
.skip:focus { left:16px; top:8px; background:var(--surface); padding:8px 12px; z-index:10; }
header.site { position:sticky; top:0; z-index:5; background:color-mix(in srgb,var(--bg) 88%,transparent);
              backdrop-filter:blur(8px); border-bottom:1px solid var(--line); }
header.site .wrap { display:flex; align-items:center; gap:16px; min-height:64px; flex-wrap:wrap; }
.brand { display:flex; align-items:center; gap:10px; font-weight:700; font-size:20px;
         color:var(--fg); text-decoration:none; }
nav.primary { margin-left:auto; display:flex; gap:20px; align-items:center; flex-wrap:wrap; }
nav.primary a { color:var(--muted); text-decoration:none; font-weight:500; }
nav.primary a:hover, nav.primary a:focus-visible { color:var(--fg); }
.btn { display:inline-block; padding:10px 18px; border-radius:10px; font-weight:600;
       text-decoration:none; border:1px solid var(--line); color:var(--fg); }
.btn.primary { background:var(--accent); color:var(--accent-ink); border-color:var(--accent); }
nav.primary a.btn.primary { color:var(--accent-ink); }
.hero { padding-top:56px; padding-bottom:24px; }
.eyebrow { color:var(--accent); font-weight:700; letter-spacing:.08em; text-transform:uppercase;
           font-size:13px; margin:0 0 12px; }
h1 { font-size:clamp(32px,5.5vw,56px); line-height:1.1; letter-spacing:-.02em; margin:0 0 16px; }
.lead { color:var(--muted); font-size:clamp(17px,2.2vw,20px); max-width:62ch; margin:0 0 28px; }
.actions { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:32px; }
figure.video { margin:0; }
video { width:100%; aspect-ratio:16/9; border-radius:16px; background:#000; display:block;
        border:1px solid var(--line); }
figcaption { color:var(--muted); font-size:14px; margin-top:10px; }
section { padding:56px 0; border-top:1px solid var(--line); }
h2 { font-size:clamp(24px,3.4vw,34px); letter-spacing:-.01em; margin:0 0 8px; }
.section-lead { color:var(--muted); margin:0 0 28px; max-width:65ch; }
.grid { display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); }
.card { background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:20px; }
.card h3 { margin:0 0 6px; font-size:18px; }
.card p { margin:0; color:var(--muted); }
ol.steps { list-style:none; padding:0; margin:0; display:grid; gap:16px;
           grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); counter-reset:step; }
ol.steps li { counter-increment:step; background:var(--surface); border:1px solid var(--line);
              border-radius:14px; padding:20px; }
ol.steps li::before { content:counter(step); display:inline-grid; place-items:center; width:32px;
              height:32px; border-radius:50%; background:var(--accent); color:var(--accent-ink);
              font-weight:700; margin-bottom:10px; }
ol.steps h3 { margin:0 0 4px; font-size:18px; }
ol.steps p { margin:0; color:var(--muted); }
details { background:var(--surface); border:1px solid var(--line); border-radius:12px;
          padding:14px 18px; margin-bottom:10px; }
summary { cursor:pointer; font-weight:600; }
details p { color:var(--muted); margin:10px 0 0; }
.transcript p { margin:0 0 10px; }
footer.site { border-top:1px solid var(--line); padding:32px 0 48px; color:var(--muted);
              font-size:14px; }
footer.site .wrap { display:flex; gap:24px; flex-wrap:wrap; justify-content:space-between; }
footer.site nav { display:flex; gap:18px; flex-wrap:wrap; }
footer.site a { color:var(--muted); }
@media (max-width:640px) { nav.primary { margin-left:0; width:100%; gap:14px; padding-bottom:10px; } }
"""


def _e(text: str) -> str:
    return html.escape(text, quote=True)


@cache
def home_page() -> str:
    url = site_url()
    features = "\n".join(
        f'<article class="card"><h3>{_e(t)}</h3><p>{_e(d)}</p></article>' for t, d in FEATURES
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
<meta name="theme-color" content="#0b1020">
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
<link rel="alternate" type="text/plain" href="/llms.txt" title="VeroniQA for language models">
<script type="application/ld+json">{ld}</script>
<style>{STYLE}</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="site">
  <div class="wrap">
    <a class="brand" href="/" aria-label="VeroniQA home">{LOGO.format(size=32)}<span>VeroniQA</span></a>
    <nav class="primary" aria-label="Main">
      <a href="#features">Features</a>
      <a href="#how-it-works">How it works</a>
      <a href="#faq">FAQ</a>
      <a href="{REPO_URL}" rel="noopener">GitHub</a>
      <a class="btn primary" href="/talk/">Talk to VeroniQA</a>
    </nav>
  </div>
</header>
<main id="main">
  <div class="wrap hero">
    <p class="eyebrow">Built on AgentQA</p>
    <h1>VeroniQA, the AI assistant for API testing</h1>
    <p class="lead">Give her your requirements, documents and links. She answers with sources,
    runs the tests and explains what she found.</p>
    <div class="actions">
      <a class="btn primary" href="/talk/">Talk to VeroniQA</a>
      <a class="btn" href="#transcript">Read the transcript</a>
    </div>
    <figure class="video">
      <video src="/videos/veroniqa-intro.mp4" poster="/videos/veroniqa-intro.jpg" controls
        preload="metadata" playsinline aria-label="Meet VeroniQA, an 85-second introduction"></video>
      <figcaption>Meet VeroniQA in 85 seconds. Captions are built in; the
      <a href="#transcript">transcript</a> is below.</figcaption>
    </figure>
  </div>

  <section id="features" aria-labelledby="features-title">
    <div class="wrap">
      <h2 id="features-title">What VeroniQA does</h2>
      <p class="section-lead">One place for an API's requirements, questions and test runs,
      on top of an engine that is observable, guarded and measured.</p>
      <div class="grid">
{features}
      </div>
    </div>
  </section>

  <section id="how-it-works" aria-labelledby="how-title">
    <div class="wrap">
      <h2 id="how-title">How it works</h2>
      <p class="section-lead">The demo starts you with a sample Orders API that has planted
      bugs, so you can try every step right away.</p>
      <ol class="steps">
{steps}
      </ol>
      <p class="actions" style="margin-top:28px"><a class="btn primary" href="/talk/">Try it now</a></p>
    </div>
  </section>

  <section id="faq" aria-labelledby="faq-title">
    <div class="wrap">
      <h2 id="faq-title">Questions</h2>
{faq}
    </div>
  </section>

  <section id="transcript" aria-labelledby="transcript-title">
    <div class="wrap transcript">
      <h2 id="transcript-title">Video transcript</h2>
{words}
    </div>
  </section>
</main>
<footer class="site">
  <div class="wrap">
    <div>
      <a class="brand" href="/" aria-label="VeroniQA home">{LOGO.format(size=24)}<span>VeroniQA</span></a>
      <p>A proof of concept built on AgentQA. The public demo uses simulated models.</p>
    </div>
    <nav aria-label="Footer">
      <a href="/">Home</a>
      <a href="/talk/">Talk to VeroniQA</a>
      <a href="{REPO_URL}" rel="noopener">Source code</a>
      <a href="{REPO_URL}/blob/main/docs/SECURITY.md" rel="noopener">Security</a>
      <a href="/llms.txt">llms.txt</a>
      <a href="/sitemap.xml">Sitemap</a>
    </nav>
  </div>
</footer>
</body>
</html>"""


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
        lines += [f"User-agent: {agent}", "Allow: /", "Disallow: /talk/_stcore/", ""]
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
</urlset>
"""


def llms_txt() -> str:
    """A plain summary for language models (the llms.txt convention)."""
    url = site_url()
    features = "\n".join(f"- {t}: {d}" for t, d in FEATURES)
    faq = "\n".join(f"- {q} {a}" for q, a in FAQ)
    return f"""# VeroniQA

> {DESCRIPTION}

VeroniQA is a proof of concept. The public demo at {url}/ uses simulated models and runs tests
only against a bundled sample Orders API with planted bugs.

## Links

- [Home page with the introduction video]({url}/)
- [Talk to VeroniQA (the app)]({url}/talk/)
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
