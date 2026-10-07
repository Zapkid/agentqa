"""Turn endpoints and Markdown requirement docs into retrievable chunks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from agentqa.models import Chunk, Endpoint


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "section"


def _schema_summary(schema: dict[str, Any] | None, depth: int = 0) -> str:
    if not schema:
        return "none"
    if "anyOf" in schema:
        return " | ".join(_schema_summary(s, depth) for s in schema["anyOf"])
    t = schema.get("type", "object")
    if t == "array":
        return f"array<{_schema_summary(schema.get('items'), depth)}>"
    if t == "object" and "properties" in schema and depth < 2:
        req = set(schema.get("required", []))
        fields = ", ".join(
            f"{k}{'*' if k in req else ''}: {_schema_summary(v, depth + 1)}"
            for k, v in schema["properties"].items()
        )
        return "{" + fields + "}"
    extra = {
        k: schema[k] for k in ("enum", "format", "minimum", "maximum", "pattern") if k in schema
    }
    return t + (json.dumps(extra) if extra else "")


def endpoint_chunk(ep: Endpoint) -> Chunk:
    lines = [f"{ep.id}", f"summary: {ep.summary or ep.operation_id}"]
    if ep.description:
        lines.append(f"description: {ep.description}")
    lines.append(
        f"auth: {'bearer (' + ','.join(ep.auth_schemes) + ')' if ep.requires_auth else 'none'}"
    )
    for p in ep.params:
        lines.append(
            f"param {p.location}.{p.name}{' (required)' if p.required else ''}: "
            f"{_schema_summary(p.schema_)} {p.description}".rstrip()
        )
    if ep.request_schema:
        lines.append(f"request body: {_schema_summary(ep.request_schema)}")
    for code, schema in sorted(ep.responses.items()):
        lines.append(f"response {code}: {_schema_summary(schema)}")
    text = "\n".join(lines)
    return Chunk(id=f"spec:{ep.id}", kind="endpoint", text=text, endpoint_id=ep.id, sha=_sha(text))


def doc_chunks(path: Path, max_chars: int = 1800) -> list[Chunk]:
    """Split a Markdown doc by headings; long sections are split by paragraph."""
    text = path.read_text(encoding="utf-8")
    doc = path.stem
    sections: list[tuple[str, str]] = []
    current_title, buf = "intro", list[str]()
    for line in text.splitlines():
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            if "".join(buf).strip():
                sections.append((current_title, "\n".join(buf).strip()))
            current_title, buf = m.group(2).strip(), [line]
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((current_title, "\n".join(buf).strip()))
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for title, body in sections:
        parts = [body] if len(body) <= max_chars else _split_paragraphs(body, max_chars)
        for i, part in enumerate(parts):
            base = f"doc:{doc}#{_slug(title)}"
            n = seen.get(base, 0)
            seen[base] = n + 1
            cid = base if n == 0 and i == 0 else f"{base}-{n + i}"
            chunks.append(
                Chunk(id=cid, kind="doc", text=part, doc=path.name, section=title, sha=_sha(part))
            )
    return chunks


def _split_paragraphs(body: str, max_chars: int) -> list[str]:
    out, buf = [], ""
    for para in body.split("\n\n"):
        if buf and len(buf) + len(para) > max_chars:
            out.append(buf.strip())
            buf = ""
        buf += para + "\n\n"
    if buf.strip():
        out.append(buf.strip())
    return out
