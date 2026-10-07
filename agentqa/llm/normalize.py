"""Normalise run-specific values out of LLM inputs.

Ports, generated ids, timestamps and temp paths change on every run. Leaving them in a prompt
makes replay mode miss the cache and wastes provider-side caching; they carry no information
the model needs. Evidence shown to people keeps the raw values.
"""

from __future__ import annotations

import re

_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"https?://(127\.0\.0\.1|localhost|\[::1\]):\d+"), "{base_url}"),
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"), "<id>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?"), "<timestamp>"),
    (re.compile(r'\\?"(ts|elapsed_ms|ms|duration_s)\\?": ?[0-9.]+,? ?'), ""),
    (re.compile(r"/tmp/\S+?/(suite|exec|verify)/"), ""),
    (re.compile(r"\bat 0x[0-9a-f]+\b"), "at <addr>"),
    (re.compile(r"agentqa-<id>"), "agentqa-<key>"),
]


def normalize(text: str) -> str:
    for pattern, repl in _RULES:
        text = pattern.sub(repl, text)
    return text
