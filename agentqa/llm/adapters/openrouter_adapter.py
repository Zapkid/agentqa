"""OpenRouter adapter (OpenAI-compatible Chat Completions via the ``openai`` SDK).

- Structured output: ``response_format = {type: json_schema, json_schema: {name, strict, schema}}``
  on models that support it; the router validates the output with pydantic regardless.
- Cached prompt tokens, when the upstream reports them, come from
  ``usage.prompt_tokens_details.cached_tokens``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agentqa import config
from agentqa.llm.types import (
    Message,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
    RawResponse,
    ToolCall,
    ToolSpec,
    Usage,
)


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        elif m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in m.tool_calls
                    ],
                }
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out


class OpenRouterAdapter:
    provider = "openrouter"

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            from openai import OpenAI

            base_url = config.providers().providers["openrouter"].base_url
            client = OpenAI(
                base_url=base_url,
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                max_retries=0,
                timeout=120.0,
            )
        self.client = client

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
        import openai

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": to_openai_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "strict": True, "schema": json_schema},
            }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except openai.RateLimitError as exc:
            raise RateLimited(str(exc)) from exc
        except (
            openai.InternalServerError,
            openai.APIConnectionError,
            openai.APITimeoutError,
        ) as exc:
            raise ProviderUnavailable(str(exc)) from exc
        except openai.APIStatusError as exc:
            if getattr(exc, "status_code", 0) >= 500:
                raise ProviderUnavailable(str(exc)) from exc
            raise ProviderError(str(exc)) from exc
        if not resp.choices:
            raise ProviderUnavailable("empty choices")
        choice = resp.choices[0]
        calls: list[ToolCall] = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        u = resp.usage
        cached = 0
        if u is not None and getattr(u, "prompt_tokens_details", None) is not None:
            cached = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
        return RawResponse(
            text=choice.message.content or "",
            tool_calls=calls,
            usage=Usage(
                input_tokens=(u.prompt_tokens if u else 0),
                output_tokens=(u.completion_tokens if u else 0),
                cached_input_tokens=cached,
            ),
            finish_reason=str(choice.finish_reason or "stop"),
            response_model=str(resp.model or model),
        )
