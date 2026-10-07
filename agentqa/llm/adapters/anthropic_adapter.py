"""Anthropic Messages API adapter (official ``anthropic`` SDK).

- Structured output: ``output_config.format = {type: json_schema, schema}``.
- Prompt caching: ``cache_control: {type: ephemeral}`` on the static system prefix.
- Usage: ``input_tokens`` (uncached) + ``cache_read_input_tokens`` + ``cache_creation_input_tokens``.
- Opus 5.5 / Sonnet 5.5 reject sampling parameters and disabled thinking, so temperature is not
  sent to them and depth is set with ``output_config.effort`` instead.
- Assistant turns are replayed with their original content blocks (thinking blocks included).
"""

from __future__ import annotations

import os
from typing import Any

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

# Models whose API accepts output_config.effort and rejects temperature (as of 2026-10-07).
EFFORT_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
# Models that support the server-side refusal fallback beta.
SERVER_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}


def to_anthropic_messages(
    messages: list[Message],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            block: dict[str, Any] = {"type": "text", "text": m.content}
            if m.cache_prefix:
                block["cache_control"] = {"type": "ephemeral"}
            system.append(block)
        elif m.role == "user":
            ublock: dict[str, Any] = {"type": "text", "text": m.content}
            if m.cache_prefix:
                ublock["cache_control"] = {"type": "ephemeral"}
            out.append({"role": "user", "content": [ublock]})
        elif m.role == "assistant":
            if m.provider_raw and m.provider_raw.get("provider") == "anthropic":
                out.append({"role": "assistant", "content": m.provider_raw["content"]})
                continue
            content: list[dict[str, Any]] = []
            if m.content:
                content.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                content.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            out.append({"role": "assistant", "content": content})
        elif m.role == "tool":
            result = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            # All tool results for one assistant turn go in a single user message.
            if (
                out
                and out[-1]["role"] == "user"
                and all(b.get("type") == "tool_result" for b in out[-1]["content"])
            ):
                out[-1]["content"].append(result)
            else:
                out.append({"role": "user", "content": [result]})
    return system, out


class AnthropicAdapter:
    provider = "anthropic"

    def __init__(self, client: Any | None = None, effort: str | None = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=0, timeout=120.0
            )
        self.client = client
        self.effort = effort or os.environ.get("AGENTQA_ANTHROPIC_EFFORT", "low")

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
        import anthropic

        system, msgs = to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": msgs}
        if system:
            kwargs["system"] = system
        output_config: dict[str, Any] = {}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if model in EFFORT_MODELS:
            output_config["effort"] = self.effort
        else:
            kwargs["temperature"] = temperature
        if output_config:
            kwargs["output_config"] = output_config
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        if model in SERVER_FALLBACK_MODELS:
            kwargs["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
            kwargs["extra_body"] = {"fallbacks": "default"}
        try:
            resp = self.client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            raise RateLimited(str(exc)) from exc
        except (
            anthropic.InternalServerError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
        ) as exc:
            raise ProviderUnavailable(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if getattr(exc, "status_code", 0) >= 500:
                raise ProviderUnavailable(str(exc)) from exc
            raise ProviderError(str(exc)) from exc

        if resp.stop_reason == "refusal":
            raise ProviderUnavailable(f"refusal: {getattr(resp, 'stop_details', None)}")
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        u = resp.usage
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        usage = Usage(
            input_tokens=u.input_tokens + cache_read + cache_write,
            output_tokens=u.output_tokens,
            cached_input_tokens=cache_read,
            cache_write_tokens=cache_write,
        )
        return RawResponse(
            text="".join(text_parts),
            tool_calls=calls,
            usage=usage,
            finish_reason=str(resp.stop_reason),
            response_model=str(resp.model),
            provider_raw={
                "provider": "anthropic",
                "content": [b.model_dump(exclude_none=True) for b in resp.content],
            },
        )
