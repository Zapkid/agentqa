"""Google Gemini adapter (official ``google-genai`` SDK).

- Structured output: ``response_mime_type=application/json`` + ``response_json_schema``.
- Prompt caching: Gemini 2.5+/3.x apply implicit caching to repeated prefixes; the cached part
  is reported in ``usage_metadata.cached_content_token_count`` and recorded separately.
- Tool calling: function declarations with ``parameters_json_schema``; automatic function
  calling is disabled (the agent loop owns tool execution).
- Gemini 3 returns thought signatures with function calls; model turns are replayed verbatim.
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


class GeminiAdapter:
    provider = "gemini"

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            from google import genai

            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.client = client

    def _contents(self, messages: list[Message]) -> tuple[str | None, list[Any]]:
        from google.genai import types

        system: list[str] = []
        contents: list[Any] = []
        for m in messages:
            if m.role == "system":
                system.append(m.content)
            elif m.role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m.content)]))
            elif m.role == "assistant":
                if m.provider_raw and m.provider_raw.get("provider") == "gemini":
                    contents.append(types.Content.model_validate(m.provider_raw["content"]))
                    continue
                parts = [types.Part(text=m.content)] if m.content else []
                for tc in m.tool_calls:
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                id=tc.id, name=tc.name, args=tc.arguments
                            )
                        )
                    )
                contents.append(types.Content(role="model", parts=parts))
            elif m.role == "tool":
                part = types.Part(
                    function_response=types.FunctionResponse(
                        id=m.tool_call_id, name=m.name or "tool", response={"result": m.content}
                    )
                )
                if (
                    contents
                    and contents[-1].role == "user"
                    and all(p.function_response is not None for p in contents[-1].parts)
                ):
                    contents[-1].parts.append(part)
                else:
                    contents.append(types.Content(role="user", parts=[part]))
        return ("\n\n".join(system) or None), contents

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
        from google.genai import errors, types

        system, contents = self._contents(messages)
        cfg: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if system:
            cfg["system_instruction"] = system
        if json_schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_json_schema"] = json_schema
        if tools:
            cfg["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t.name,
                            description=t.description,
                            parameters_json_schema=t.parameters,
                        )
                        for t in tools
                    ]
                )
            ]
        try:
            resp = self.client.models.generate_content(
                model=model, contents=contents, config=types.GenerateContentConfig(**cfg)
            )
        except errors.APIError as exc:
            code = int(getattr(exc, "code", 0) or 0)
            if code == 429:
                raise RateLimited(str(exc)) from exc
            if code >= 500 or code == 0:
                raise ProviderUnavailable(str(exc)) from exc
            raise ProviderError(str(exc)) from exc

        calls = [
            ToolCall(id=fc.id or f"call_{i}", name=fc.name or "", arguments=dict(fc.args or {}))
            for i, fc in enumerate(resp.function_calls or [])
        ]
        um = resp.usage_metadata
        prompt_tokens = (um.prompt_token_count or 0) if um else 0
        output = ((um.candidates_token_count or 0) + (um.thoughts_token_count or 0)) if um else 0
        cached = (um.cached_content_token_count or 0) if um else 0
        candidate = resp.candidates[0] if resp.candidates else None
        finish = str(candidate.finish_reason) if candidate and candidate.finish_reason else "stop"
        raw_content = (
            candidate.content.model_dump(exclude_none=True, mode="json")
            if (candidate and candidate.content)
            else None
        )
        text = ""
        if candidate and candidate.content and candidate.content.parts:
            text = "".join(p.text for p in candidate.content.parts if p.text and not p.thought)
        return RawResponse(
            text=text,
            tool_calls=calls,
            usage=Usage(
                input_tokens=prompt_tokens, output_tokens=output, cached_input_tokens=cached
            ),
            finish_reason=finish,
            response_model=str(getattr(resp, "model_version", None) or model),
            provider_raw={"provider": "gemini", "content": raw_content} if raw_content else None,
        )
