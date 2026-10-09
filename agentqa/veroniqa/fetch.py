"""Fetch a linked web page for the knowledge base, safely.

Links come from users and documents, so fetching one must not become a way to reach internal
services (server-side request forgery). Rules: http(s) only; no credentials in the URL; every
address the host resolves to must be public (no loopback, private, link-local, reserved or
multicast ranges), checked again after each redirect; at most 3 redirects; 2 MB; text types
only. Set AGENTQA_ALLOW_PRIVATE_LINKS=1 to link to a docs server on your own network.

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

MAX_BYTES = 2_000_000
MAX_REDIRECTS = 3
TIMEOUT_S = 10.0
TEXT_TYPES = ("text/html", "text/plain", "text/markdown", "application/xhtml+xml")

Resolver = Callable[[str, int], list[str]]


class FetchError(ValueError):
    pass


@dataclass
class FetchedPage:
    url: str
    title: str
    text: str


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


def fetch_url(
    url: str, *, client: httpx.Client | None = None, resolver: Resolver = _resolve
) -> FetchedPage:
    own = client is None
    http = client or httpx.Client(timeout=TIMEOUT_S, headers={"User-Agent": "AgentQA-Veroniqa"})
    try:
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
                if ctype and ctype not in TEXT_TYPES:
                    raise FetchError(f"unsupported content type {ctype}")
                body = bytearray()
                for chunk in resp.iter_bytes():
                    body += chunk
                    if len(body) > MAX_BYTES:
                        raise FetchError("the page is larger than 2 MB")
                text = body.decode(resp.encoding or "utf-8", errors="replace")
                if ctype in ("text/html", "application/xhtml+xml") or text.lstrip()[:1] == "<":
                    title, text = html_to_text(text)
                else:
                    title = ""
                return FetchedPage(url=current, title=title or current, text=text)
        raise FetchError("too many redirects")
    except httpx.HTTPError as exc:
        raise FetchError(f"could not fetch the page ({type(exc).__name__})") from exc
    finally:
        if own:
            http.close()
