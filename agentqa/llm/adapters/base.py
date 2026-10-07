"""Adapter protocol: one provider, one HTTP round-trip, no retries, no caching."""

from __future__ import annotations

import json
from typing import Any, Protocol

from agentqa.llm.types import Message, RawResponse, ToolSpec


class ProviderAdapter(Protocol):
    provider: str

    def raw_complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        json_schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> RawResponse: ...


def estimate_tokens(text: str) -> int:
    """Rough estimate (~4 chars/token) used for rate-limit pre-checks and the simulator."""
    return max(1, len(text) // 4)


def messages_text(messages: list[Message]) -> str:
    parts = []
    for m in messages:
        parts.append(m.content)
        for tc in m.tool_calls:
            parts.append(json.dumps(tc.arguments))
    return "\n".join(parts)


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a pydantic JSON schema with ``additionalProperties: false`` on every
    object, which strict structured-output modes require."""

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            node = {k: walk(v) for k, v in node.items()}
            if node.get("type") == "object" and "additionalProperties" not in node:
                node["additionalProperties"] = False
            return node
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    result: dict[str, Any] = walk(schema)
    return result
