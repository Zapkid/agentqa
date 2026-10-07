"""Scripted fake adapter for unit tests: returns queued responses (or raises queued errors)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentqa.llm.adapters.base import estimate_tokens, messages_text
from agentqa.llm.types import Message, RawResponse, ToolSpec, Usage

Script = RawResponse | Exception | Callable[[list[Message]], RawResponse]


class FakeAdapter:
    def __init__(self, provider: str = "simulated", script: list[Script] | None = None) -> None:
        self.provider = provider
        self.script: list[Script] = list(script or [])
        self.calls: list[dict[str, Any]] = []

    def queue(self, *items: Script) -> None:
        self.script.extend(items)

    def raw_complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        json_schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> RawResponse:
        self.calls.append(
            {"model": model, "messages": messages, "tools": tools, "schema": json_schema}
        )
        if not self.script:
            raise AssertionError("FakeAdapter script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(messages)
        if item.usage.total == 0:
            item = item.model_copy(
                update={
                    "usage": Usage(
                        input_tokens=estimate_tokens(messages_text(messages)),
                        output_tokens=estimate_tokens(item.text),
                    )
                }
            )
        return item.model_copy(update={"response_model": item.response_model or model})
