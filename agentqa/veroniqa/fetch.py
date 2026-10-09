"""Fetch a linked web page for the knowledge base, safely.

Links come from users and documents, so fetching one must not become a way to reach internal
services (server-side request forgery). Rules: http(s) only; no credentials in the URL; every
address the host resolves to must be public (no loopback, private, link-local, reserved or
multicast ranges), checked again after each redirect; at most 3 redirects; 2 MB; text types
only. Set AGENTQA_ALLOW_PRIVATE_LINKS=1 to link to a docs server on your own network.

API documentation is understood: an OpenAPI or Swagger file (JSON or YAML) becomes a readable
description of every endpoint, and a Swagger UI or ReDoc page, which has no text of its own, is
followed to the spec it loads (apidocs.py). Those extra requests pass the same checks.

Residual risk: the address is checked before the request and resolved again by the HTTP client,
so a DNS answer that changes in between (rebinding) is not covered. Docs/SECURITY.md lists it.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, ClassVar
from urllib.parse import urljoin, urlsplit

import httpx

from agentqa.veroniqa import apidocs

MAX_BYTES = 2_000_000
MAX_REDIRECTS = 3
TIMEOUT_S = 10.0
HTML_TYPES = ("text/html", "application/xhtml+xml")
SPEC_TYPES = (
    "application/json",
    "application/yaml",
    "application/x-yaml",
    "text/yaml",
    "text/x-yaml",
    "application/vnd.oai.openapi",
    "application/vnd.oai.openapi+json",
)
TEXT_TYPES = (*HTML_TYPES, "text/plain", "text/markdown", *SPEC_TYPES)
SCRIPT_TYPES = ("application/javascript", "text/javascript", "application/x-javascript")
MAX_SPEC_PROBES = 8  # extra requests a docs page may cost while looking for its spec

Resolver = Callable[[str, int], list[str]]


class FetchError(ValueError):
    pass


@dataclass
class FetchedPage:
    url: str
    title: str
    text: str
    spec_url: str | None = None  # set when the link was (or led to) an OpenAPI/Swagger spec


def _resolve(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    return sorted({str(info[4][0]) for info in infos})


def check_url(url: str, resolver: Resolver = _resolve) -> None:
    """Raise FetchError unless the URL is http(s) and resolves only to public addresses."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise FetchError("only http and https links are allowed")
    if parts.username or parts.password:
        raise FetchError("links with credentials are not allowed")
    if not parts.hostname:
        raise FetchError("the link has no host")
    if os.environ.get("AGENTQA_ALLOW_PRIVATE_LINKS") == "1":
        return
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        addresses = resolver(parts.hostname, port)
    except OSError as exc:
        raise FetchError(f"cannot resolve {parts.hostname}") from exc
    if not addresses:
        raise FetchError(f"cannot resolve {parts.hostname}")
    for addr in addresses:
        if not ipaddress.ip_address(addr.split("%")[0]).is_global:
            raise FetchError(f"{parts.hostname} resolves to a non-public address ({addr})")


class _TextExtractor(HTMLParser):
    SKIP: ClassVar[frozenset[str]] = frozenset(
        {"script", "style", "noscript", "template", "svg", "head"}
    )
    BLOCK: ClassVar[frozenset[str]] = frozenset(
        {"p", "div", "li", "br", "tr", "section", "article", "h1", "h2", "h3", "h4", "pre"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in ("h1", "h2", "h3"):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data.strip()
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    """(title, text) of an HTML page: visible text only, headings kept as Markdown headings."""
    p = _TextExtractor()
    p.feed(html)
    lines = [" ".join(line.split()) for line in "".join(p.parts).splitlines()]
    text, blank = [], False
    for line in lines:
        if line or not blank:
            text.append(line)
        blank = not line
    return p.title, "\n".join(text).strip()


def _download(
    url: str, http: httpx.Client, resolver: Resolver, types: tuple[str, ...] = TEXT_TYPES
) -> tuple[str, str, str]:
    """(final URL, content type, text) of one checked request, following checked redirects."""
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        check_url(current, resolver)
        with http.stream("GET", current, follow_redirects=False) as resp:
            if resp.is_redirect:
                location = resp.headers.get("location")
                if not location:
                    raise FetchError("redirect without a location")
                current = urljoin(current, location)
                continue
            if resp.status_code >= 400:
                raise FetchError(f"the page returned HTTP {resp.status_code}")
            ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if ctype and ctype not in types:
                raise FetchError(f"unsupported content type {ctype}")
            body = bytearray()
            for chunk in resp.iter_bytes():
                body += chunk
                if len(body) > MAX_BYTES:
                    raise FetchError("the page is larger than 2 MB")
            return current, ctype, body.decode(resp.encoding or "utf-8", errors="replace")
    raise FetchError("too many redirects")


def _spec_page(url: str, text: str) -> FetchedPage | None:
    spec = apidocs.parse_spec(text)
    if spec is None:
        return None
    title, markdown = apidocs.spec_to_markdown(spec, url)
    return FetchedPage(url=url, title=title, text=markdown, spec_url=url)


def _find_spec(
    html: str, page_url: str, http: httpx.Client, resolver: Resolver
) -> FetchedPage | None:
    """Follow a documentation page to the spec it displays: addresses named in the page, then in
    its swagger-initializer.js, then the usual locations next to the page."""
    urls = apidocs.spec_urls_in(html, page_url)
    budget = MAX_SPEC_PROBES
    script = apidocs.initializer_url(html, page_url)
    if script:
        budget -= 1
        try:
            final, _, js = _download(script, http, resolver, (*SCRIPT_TYPES, "text/plain"))
            urls += apidocs.spec_urls_in(js, final)
        except FetchError:
            pass
    if apidocs.looks_like_docs_ui(html):
        urls += apidocs.candidate_spec_urls(page_url)
    for candidate in list(dict.fromkeys(urls))[: max(budget, 0)]:
        try:
            final, ctype, text = _download(candidate, http, resolver)
        except FetchError:
            continue
        if ctype not in HTML_TYPES and (page := _spec_page(final, text)):
            return page
    return None


def fetch_url(
    url: str, *, client: httpx.Client | None = None, resolver: Resolver = _resolve
) -> FetchedPage:
    own = client is None
    http = client or httpx.Client(timeout=TIMEOUT_S, headers={"User-Agent": "AgentQA-VeroniQA"})
    try:
        final, ctype, text = _download(url, http, resolver)
        is_html = ctype in HTML_TYPES or text.lstrip()[:1] == "<"
        if not is_html:
            return _spec_page(final, text) or FetchedPage(url=final, title=final, text=text)
        title, body = html_to_text(text)
        if apidocs.looks_like_docs_ui(text) or not body.strip():
            if found := _find_spec(text, final, http, resolver):
                return found
        if not body.strip():
            raise FetchError(
                "the page has no readable text (it is probably built with JavaScript). Link to "
                "the document itself; for an API, link to its OpenAPI or Swagger file"
            )
        return FetchedPage(url=final, title=title or final, text=body)
    except httpx.HTTPError as exc:
        raise FetchError(f"could not fetch the page ({type(exc).__name__})") from exc
    finally:
        if own:
            http.close()
