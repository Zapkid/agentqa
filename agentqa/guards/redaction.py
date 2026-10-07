"""Secret and PII redaction used by logs, traces, request logs, reports and generated code.

What it stops: an API key, bearer token, HMAC secret or an e-mail address from the target API
ending up in a log line, a span attribute, a committed report or a generated test file.
"""

from __future__ import annotations

import re
from typing import Any

from agentqa.obs import metrics

REDACTED = "[REDACTED]"

# Order matters: specific key formats first, generic patterns last.
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}")),
    ("openrouter_key", re.compile(r"sk-or-[A-Za-z0-9_\-]{10,}")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_\-]{30,}")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer", re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}")),
    (
        "kv_secret",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|password|passwd|token)(\"?\s*[:=]\s*\"?)([^\s\"',}]{6,})"
        ),
    ),
]
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("card_number", re.compile(r"\b(?:\d[ \-]?){13,16}\b")),
]

SENSITIVE_KEYS = {
    "authorization",
    "x-api-key",
    "api_key",
    "apikey",
    "password",
    "secret",
    "x-signature",
    "cookie",
    "set-cookie",
    "token",
}


def redact_text(text: str, *, pii: bool = True) -> str:
    out = text
    for name, pattern in SECRET_PATTERNS:
        if name == "bearer":
            out = pattern.sub(lambda m: m.group(1) + REDACTED, out)
        elif name == "kv_secret":
            out = pattern.sub(lambda m: m.group(1) + m.group(2) + REDACTED, out)
        else:
            out = pattern.sub(REDACTED, out)
    if pii:
        for _, pattern in PII_PATTERNS:
            out = pattern.sub(REDACTED, out)
    return out


def redact(value: Any, *, pii: bool = True) -> Any:
    """Recursively redact strings, and whole values under sensitive keys."""
    if isinstance(value, str):
        return redact_text(value, pii=pii)
    if isinstance(value, dict):
        return {
            k: (REDACTED if str(k).lower() in SENSITIVE_KEYS and v else redact(v, pii=pii))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v, pii=pii) for v in value]
    if isinstance(value, tuple):
        return tuple(redact(v, pii=pii) for v in value)
    return value


def find_secrets(text: str) -> list[str]:
    """Names of secret patterns present in ``text`` (used by the generated-code secret scan)."""
    hits = [
        name for name, pattern in SECRET_PATTERNS if name != "kv_secret" and pattern.search(text)
    ]
    if hits:
        metrics.inc(
            "agentqa_guardrail_events_total", guardrail="output_safety", action="secret_found"
        )
    return hits
