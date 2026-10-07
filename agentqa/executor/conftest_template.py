"""conftest.py for generated suites (copied verbatim into each suite directory).

Fixtures available to generated tests (this is what the generator's ``list_fixtures`` tool returns):

- ``client``: ``httpx.Client`` bound to the target ``base_url`` (sandboxed, logged)
- ``auth``: dict role -> headers, e.g. ``auth["admin"]``, ``auth["customer"]``, ``auth["other_customer"]``
- ``customer_id`` / ``other_customer_id``: ids of the customer roles
- ``sign_webhook(body: bytes) -> str``: signature header value for the payment webhook
- ``webhook_header``: name of the signature header
- ``find_id(list_path, headers, limit=1, predicate=None)``: ids from a collection endpoint
- ``schema_check(data, schema)``: strict JSON-schema check (no undocumented properties)

Sandbox (guardrail 3): only the target host is reachable (enforced on the HTTP client *and*
on raw sockets), and unless the target is a declared sandbox only read-only methods are allowed.
Every request/response is logged (redacted) per test; blocked attempts are guardrail events.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest

CONFIG: dict[str, Any] = json.loads(Path(os.environ["AGENTQA_SUITE_CONFIG"]).read_text())
OUT = Path(CONFIG["out_dir"])
BASE = urlparse(CONFIG["base_url"])
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _allowed(host: str | None, port: int | None) -> bool:
    if port != BASE.port:
        return False
    return host == BASE.hostname or (host in LOOPBACK and BASE.hostname in LOOPBACK)


READ_ONLY = {"GET", "HEAD", "OPTIONS"}
_current_test: dict[str, str] = {"name": "setup"}

try:  # redaction from the main package when importable
    from agentqa.guards.redaction import redact
except ImportError:  # pragma: no cover

    def redact(value: Any, *, pii: bool = True) -> Any:
        return value


class SandboxViolation(Exception):
    pass


def _event(kind: str, detail: str) -> None:
    with open(OUT / "guardrail_events.jsonl", "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "ts": time.time(),
                    "guardrail": "sandbox",
                    "action": "block",
                    "kind": kind,
                    "detail": detail,
                    "test": _current_test["name"],
                }
            )
            + "\n"
        )


# --- network allowlist at the socket layer (defence in depth) --------------------------------
_real_connect = socket.socket.connect


def _guarded_connect(self: socket.socket, address: Any) -> Any:
    if isinstance(address, tuple) and len(address) >= 2:
        host, port = address[0], address[1]
        if not _allowed(host, port):
            _event("host", f"{host}:{port}")
            raise SandboxViolation(f"network access to {host}:{port} is not allowed")
    return _real_connect(self, address)


socket.socket.connect = _guarded_connect  # type: ignore[method-assign,assignment]


class SandboxTransport(httpx.BaseTransport):
    def __init__(self) -> None:
        self.inner = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        if not _allowed(url.host, url.port):
            _event("host", str(url))
            raise SandboxViolation(f"host {url.host}:{url.port} is not the target")
        if request.method not in READ_ONLY and not CONFIG["allow_mutations"]:
            _event("method", f"{request.method} {url.path}")
            raise SandboxViolation(f"{request.method} is blocked: target is not a declared sandbox")
        if request.method in CONFIG.get("blocked_methods", []):
            _event("method", f"{request.method} {url.path}")
            raise SandboxViolation(f"{request.method} is blocked by policy")
        t0 = time.perf_counter()
        response = self.inner.handle_request(request)
        response.read()
        body = request.content.decode("utf-8", "replace") if request.content else None
        record = {
            "test": _current_test["name"],
            "method": request.method,
            "url": str(url),
            "path": url.path,
            "status": response.status_code,
            "request_headers": dict(request.headers),
            "request_body": body,
            "response_body": response.text[:4000],
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "ts": time.time(),
        }
        with open(OUT / "exchanges.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(redact(record)) + "\n")
        return response


@pytest.fixture
def base_url() -> str:
    return str(CONFIG["base_url"])


@pytest.fixture
def client() -> Iterator[httpx.Client]:
    with httpx.Client(base_url=CONFIG["base_url"], transport=SandboxTransport(), timeout=15) as c:
        yield c


@pytest.fixture
def auth() -> dict[str, dict[str, str]]:
    return {role: dict(h) for role, h in CONFIG["auth"].items()}


@pytest.fixture
def customer_id() -> str | None:
    return CONFIG.get("customer_id")


@pytest.fixture
def other_customer_id() -> str | None:
    return CONFIG.get("other_customer_id")


@pytest.fixture
def webhook_header() -> str:
    return str(CONFIG.get("webhook_header", "X-Signature"))


@pytest.fixture
def sign_webhook() -> Callable[[bytes], str]:
    secret = str(CONFIG.get("webhook_secret", "")).encode()

    def sign(body: bytes) -> str:
        return hmac.new(secret, body, hashlib.sha256).hexdigest()

    return sign


@pytest.fixture
def find_id(client: httpx.Client) -> Callable[..., list[str]]:
    def find(
        list_path: str,
        headers: dict[str, str],
        limit: int = 1,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[str]:
        body = client.get(list_path, headers=headers).json()
        items = body.get("items", []) if isinstance(body, dict) else body
        out = [
            i["id"]
            for i in items
            if isinstance(i, dict) and "id" in i and (predicate is None or predicate(i))
        ]
        return out[:limit]

    return find


def _check(data: Any, schema: dict[str, Any], path: str = "$") -> None:
    if "anyOf" in schema:
        errors = []
        for option in schema["anyOf"]:
            try:
                _check(data, option, path)
                return
            except AssertionError as exc:
                errors.append(str(exc))
        raise AssertionError(f"{path}: matches no anyOf option ({'; '.join(errors)[:300]})")
    t = schema.get("type")
    if data is None:
        assert t == "null" or schema.get("nullable"), f"{path}: null not allowed"
        return
    if t == "object" or "properties" in schema:
        assert isinstance(data, dict), f"{path}: expected object"
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            assert req in data, f"{path}: missing required property {req!r}"
        extra = set(data) - set(props)
        if extra and props and schema.get("additionalProperties") in (None, False):
            raise AssertionError(f"{path}: undocumented properties {sorted(extra)}")
        for k, v in data.items():
            if k in props:
                _check(v, props[k], f"{path}.{k}")
    elif t == "array":
        assert isinstance(data, list), f"{path}: expected array"
        for i, v in enumerate(data[:50]):
            _check(v, schema.get("items", {}), f"{path}[{i}]")
    elif t == "string":
        assert isinstance(data, str), f"{path}: expected string"
        if "enum" in schema:
            assert data in schema["enum"], f"{path}: {data!r} not in enum"
    elif t == "integer":
        assert isinstance(data, int) and not isinstance(data, bool), f"{path}: expected integer"
    elif t == "number":
        assert isinstance(data, (int, float)) and not isinstance(data, bool), (
            f"{path}: expected number"
        )
    elif t == "boolean":
        assert isinstance(data, bool), f"{path}: expected boolean"


@pytest.fixture
def schema_check() -> Callable[[Any, dict[str, Any]], None]:
    return _check


# --- result capture ---------------------------------------------------------------------------


def pytest_runtest_setup(item: pytest.Item) -> None:
    _current_test["name"] = item.name


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        outcome: str = report.outcome
        message = ""
        if report.longrepr is not None:
            message = str(getattr(report.longrepr, "reprcrash", None) or report.longrepr)[-1500:]
            if report.when == "setup" and outcome == "failed":
                outcome = "error"
            if "SandboxViolation" in str(report.longrepr):
                outcome = "blocked"
        if outcome == "skipped" and isinstance(report.longrepr, tuple):
            message = str(report.longrepr[2])
        with open(OUT / "results.jsonl", "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "test": report.nodeid.split("::")[-1],
                        "outcome": outcome,
                        "duration_s": round(report.duration, 4),
                        "message": message,
                        "ts": time.time(),
                    }
                )
                + "\n"
            )


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.failed and report.nodeid.endswith(".py"):
        with open(OUT / "results.jsonl", "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "test": Path(report.nodeid).stem,
                        "outcome": "error",
                        "duration_s": 0.0,
                        "message": f"collection error: {str(report.longrepr)[-800:]}",
                        "ts": time.time(),
                    }
                )
                + "\n"
            )
