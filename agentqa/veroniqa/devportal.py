"""The developer portal at /developers (HTML and Markdown), generated from the public API itself:
the endpoint list comes from the live OpenAPI document and the error codes from the API's own
table, so the guide cannot drift from the API."""

from __future__ import annotations

import html
import json
from functools import cache

from agentqa.veroniqa import public_api, site

CLI_GIT = "git+https://github.com/Zapkid/agentqa#subdirectory=clients/veroniqa-cli"


def _e(text: str) -> str:
    return html.escape(text, quote=True)


def endpoints() -> list[tuple[str, str, str, str]]:
    """(method, path, summary, description) for every operation in the OpenAPI document."""
    doc = public_api.openapi_document()
    out = []
    for path, item in doc["paths"].items():
        for method, op in item.items():
            out.append(
                (method.upper(), f"/api{path}", op.get("summary", ""), op.get("description", ""))
            )
    return out


def _examples() -> dict[str, str]:
    url = site.site_url()
    return {
        "status": f"curl {url}/api/v1/status",
        "ask": (
            f"curl -X POST {url}/api/v1/ask \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            '  -d \'{"question": "What is the p95 latency target for the order list?"}\''
        ),
        "cli": (
            f'pipx install "{CLI_GIT}"\n'
            "veroniqa status\n"
            'veroniqa ask "What is the p95 latency target for the order list?"\n'
            "veroniqa results --json"
        ),
        "uvx": f'uvx --from "{CLI_GIT}" veroniqa status',
        "problem": json.dumps(
            {
                "type": f"{url}/developers#error-invalid_request",
                "title": "Invalid request",
                "status": 422,
                "detail": "The request body is not valid.",
                "instance": "/api/v1/ask",
                "code": "invalid_request",
                "hint": public_api.ERRORS["invalid_request"][2],
                "errors": [{"field": "question", "message": "Field required"}],
            },
            indent=2,
        ),
    }


def _limits() -> list[tuple[str, int, int]]:
    return [(b, q, w) for b, (q, w) in public_api.LIMITS.limits.items()]


@cache
def developers_markdown() -> str:
    url = site.site_url()
    ex = _examples()
    rows = "\n".join(f"| `{m}` | `{p}` | {s} |" for m, p, s, _ in endpoints())
    errors = "\n".join(
        f"| `{code}` | {status} | {title} | {hint} |"
        for code, (status, title, hint) in public_api.ERRORS.items()
    )
    limits = "\n".join(f"- `{b}`: {q} requests per {w} seconds per client" for b, q, w in _limits())
    return f"""# VeroniQA for developers

VeroniQA's public API is a free, keyless sandbox: ask questions about the demo API and get cited
answers, read the measured results, and fetch the demo API's OpenAPI spec. Models are simulated,
so answers are extracts from the documents. Nothing you send is stored.

- Base URL: `{url}/api`
- OpenAPI 3.1: [{url}/openapi.json]({url}/openapi.json) (also [YAML]({url}/api/openapi.yaml))
- API catalog (RFC 9727): [{url}/.well-known/api-catalog]({url}/.well-known/api-catalog)
- Official CLI: `veroniqa` (Python, no dependencies)

## Quickstart

1. Check the service: `{ex["status"]}`
2. Ask a question:

```bash
{ex["ask"]}
```

3. Or use the CLI:

```bash
{ex["cli"]}
```

Run it once without installing: `{ex["uvx"]}`

## Authentication and API keys

None. The public API is a sandbox for the demo, so there are no accounts or keys; each client is
rate limited instead. To test your own API with real models, run VeroniQA yourself from
[the repository]({site.REPO_URL}).

## Endpoints

| method | path | what it does |
|---|---|---|
{rows}

Request and response schemas are in the [OpenAPI description]({url}/openapi.json).

## Errors

Every error under `/api` is an [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457) problem
document, served as `application/problem+json`, with a stable `code` and a `hint`:

```json
{ex["problem"]}
```

| code | status | title | how to fix |
|---|---|---|---|
{errors}

## Rate limits

{limits}

Every response carries `RateLimit-Policy` and `RateLimit` headers
([draft-ietf-httpapi-ratelimit-headers](https://datatracker.ietf.org/doc/draft-ietf-httpapi-ratelimit-headers/)),
for example `RateLimit: "default";r=59;t=60`. A `429 rate_limited` response adds `Retry-After`.

## Sandbox

- The interactive app: [{url}/talk/]({url}/talk/) (no sign-up; a private, temporary workspace)
- The demo API that VeroniQA tests: [its OpenAPI spec]({url}/api/v1/demo/openapi.json)

## Versioning

The API is versioned in the path (`/v1`); the current release is {public_api.API_VERSION}.
Breaking changes get a new version path.

## Machine-readable files

- [openapi.json]({url}/openapi.json), [openapi.yaml]({url}/openapi.yaml)
- [api-catalog]({url}/.well-known/api-catalog)
- [llms.txt]({url}/llms.txt)
- [sitemap.xml]({url}/sitemap.xml)
"""


@cache
def developers_page() -> str:
    url = site.site_url()
    ex = _examples()
    rows = "\n".join(
        f"<tr><td><code>{m}</code></td><td><code>{_e(p)}</code></td><td>{_e(s)}</td></tr>"
        for m, p, s, _ in endpoints()
    )
    errors = "\n".join(
        f'<tr id="error-{code}"><td><code>{code}</code></td><td>{status}</td>'
        f"<td>{_e(title)}</td><td>{_e(hint)}</td></tr>"
        for code, (status, title, hint) in public_api.ERRORS.items()
    )
    limits = "".join(
        f"<li><code>{b}</code>: {q} requests per {w} seconds per client</li>"
        for b, q, w in _limits()
    )
    body = f"""<p class="eyebrow">Developers</p>
<h1>VeroniQA for developers</h1>
<p>VeroniQA's public API is a free, keyless sandbox: ask questions about the demo API and get cited
answers, read the measured results, and fetch the demo API's OpenAPI spec. Models are simulated, so
answers are extracts from the documents. Nothing you send is stored.</p>
<ul>
<li>Base URL: <code>{url}/api</code></li>
<li>OpenAPI 3.1: <a href="/openapi.json">/openapi.json</a> (also <a href="/api/openapi.yaml">YAML</a>)</li>
<li>API catalog (RFC 9727): <a href="/.well-known/api-catalog">/.well-known/api-catalog</a></li>
<li>Official CLI: <code>veroniqa</code> (Python, no dependencies)</li>
</ul>

<h2 id="quickstart">Quickstart</h2>
<p>Check the service, then ask a question:</p>
<pre><code>{_e(ex["status"])}

{_e(ex["ask"])}</code></pre>
<p>Or use the CLI:</p>
<pre><code>{_e(ex["cli"])}</code></pre>
<p>Run it once without installing: <code>{_e(ex["uvx"])}</code></p>

<h2 id="authentication">Authentication and API keys</h2>
<p>None. The public API is a sandbox for the demo, so there are no accounts or keys; each client is
rate limited instead. To test your own API with real models, run VeroniQA yourself from
<a href="{site.REPO_URL}" rel="noopener">the repository</a>.</p>

<h2 id="endpoints">Endpoints</h2>
<div class="table"><table>
<thead><tr><th>method</th><th>path</th><th>what it does</th></tr></thead>
<tbody>
{rows}
</tbody></table></div>
<p>Request and response schemas are in the <a href="/openapi.json">OpenAPI description</a>.</p>

<h2 id="errors">Errors</h2>
<p>Every error under <code>/api</code> is an
<a href="https://www.rfc-editor.org/rfc/rfc9457" rel="noopener">RFC 9457</a> problem document,
served as <code>application/problem+json</code>, with a stable <code>code</code> and a
<code>hint</code>:</p>
<pre><code>{_e(ex["problem"])}</code></pre>
<div class="table"><table>
<thead><tr><th>code</th><th>status</th><th>title</th><th>how to fix</th></tr></thead>
<tbody>
{errors}
</tbody></table></div>

<h2 id="rate-limits">Rate limits</h2>
<ul>{limits}</ul>
<p>Every response carries <code>RateLimit-Policy</code> and <code>RateLimit</code> headers, for
example <code>RateLimit: "default";r=59;t=60</code>. A <code>429 rate_limited</code> response adds
<code>Retry-After</code>.</p>

<h2 id="sandbox">Sandbox</h2>
<ul>
<li>The interactive app: <a href="/talk/">/talk/</a> (no sign-up; a private, temporary workspace)</li>
<li>The demo API that VeroniQA tests: <a href="/api/v1/demo/openapi.json">its OpenAPI spec</a></li>
</ul>

<h2 id="versioning">Versioning</h2>
<p>The API is versioned in the path (<code>/v1</code>); the current release is
{public_api.API_VERSION}. Breaking changes get a new version path.</p>

<h2 id="machine-readable">Machine-readable files</h2>
<ul>
<li><a href="/openapi.json">openapi.json</a>, <a href="/openapi.yaml">openapi.yaml</a></li>
<li><a href="/.well-known/api-catalog">api-catalog</a></li>
<li><a href="/llms.txt">llms.txt</a>, <a href="/sitemap.xml">sitemap.xml</a></li>
</ul>"""
    return site._shell(
        "Developers · VeroniQA API, CLI and OpenAPI",
        "VeroniQA's public API: a keyless sandbox with an OpenAPI 3.1 description, RFC 9457 "
        "errors, rate-limit headers and an official CLI.",
        "/developers",
        body,
    )
