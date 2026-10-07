"""Adapters tested against stub SDK clients (no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agentqa.llm.adapters.anthropic_adapter import AnthropicAdapter, to_anthropic_messages
from agentqa.llm.adapters.gemini_adapter import GeminiAdapter
from agentqa.llm.adapters.openrouter_adapter import OpenRouterAdapter, to_openai_messages
from agentqa.llm.types import Message, ProviderError, RateLimited, ToolCall, ToolSpec

TOOLS = [
    ToolSpec(
        name="get_endpoint_schema",
        description="d",
        parameters={"type": "object", "properties": {"id": {"type": "string"}}},
    )
]
CONVO = [
    Message(role="system", content="S", cache_prefix=True),
    Message(role="user", content="U"),
    Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="t1", name="get_endpoint_schema", arguments={"id": "x"})],
    ),
    Message(role="tool", content="{}", tool_call_id="t1", name="get_endpoint_schema"),
]


class Block(SimpleNamespace):
    def model_dump(self, **_: Any) -> dict[str, Any]:
        return dict(self.__dict__)


class StubAnthropic:
    def __init__(self, resp: Any = None, exc: Exception | None = None) -> None:
        self.kwargs: dict[str, Any] = {}
        self.resp, self.exc = resp, exc
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return self.resp


def test_anthropic_message_conversion() -> None:
    system, msgs = to_anthropic_messages(CONVO)
    assert system == [{"type": "text", "text": "S", "cache_control": {"type": "ephemeral"}}]
    assert msgs[1]["content"][0]["type"] == "tool_use"
    assert msgs[2]["content"][0] == {"type": "tool_result", "tool_use_id": "t1", "content": "{}"}


def test_anthropic_request_and_usage() -> None:
    resp = SimpleNamespace(
        content=[
            Block(type="text", text='{"a":1}'),
            Block(type="tool_use", id="c", name="n", input={"q": 1}),
        ],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=100,
            cache_creation_input_tokens=0,
        ),
        stop_reason="end_turn",
        model="claude-opus-5-5",
    )
    stub = StubAnthropic(resp)
    out = AnthropicAdapter(client=stub).raw_complete(
        model="claude-opus-5-5",
        messages=CONVO,
        tools=TOOLS,
        json_schema={"type": "object"},
        temperature=0.0,
        max_tokens=100,
    )
    assert "temperature" not in stub.kwargs  # rejected by Opus 5.5
    assert stub.kwargs["output_config"]["format"]["type"] == "json_schema"
    assert stub.kwargs["output_config"]["effort"] == "low"
    assert stub.kwargs["tools"][0]["input_schema"]["type"] == "object"
    assert out.usage.input_tokens == 110 and out.usage.cached_input_tokens == 100
    assert out.tool_calls[0].arguments == {"q": 1}
    assert out.provider_raw and out.provider_raw["provider"] == "anthropic"


def test_anthropic_haiku_gets_temperature() -> None:
    resp = SimpleNamespace(
        content=[Block(type="text", text="x")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
        model="claude-haiku-4-5",
    )
    stub = StubAnthropic(resp)
    AnthropicAdapter(client=stub).raw_complete(
        model="claude-haiku-4-5",
        messages=CONVO[:2],
        tools=None,
        json_schema=None,
        temperature=0.3,
        max_tokens=10,
    )
    assert stub.kwargs["temperature"] == 0.3 and "output_config" not in stub.kwargs


def test_anthropic_errors_map() -> None:
    import anthropic
    import httpx2

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    rate = anthropic.RateLimitError("slow", response=httpx2.Response(429, request=req), body=None)
    with pytest.raises(RateLimited):
        AnthropicAdapter(client=StubAnthropic(exc=rate)).raw_complete(
            model="claude-haiku-4-5",
            messages=CONVO[:2],
            tools=None,
            json_schema=None,
            temperature=0,
            max_tokens=5,
        )
    bad = anthropic.BadRequestError("bad", response=httpx2.Response(400, request=req), body=None)
    with pytest.raises(ProviderError):
        AnthropicAdapter(client=StubAnthropic(exc=bad)).raw_complete(
            model="claude-haiku-4-5",
            messages=CONVO[:2],
            tools=None,
            json_schema=None,
            temperature=0,
            max_tokens=5,
        )


class StubOpenAI:
    def __init__(self, resp: Any) -> None:
        self.resp = resp
        self.kwargs: dict[str, Any] = {}
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.resp


def test_openrouter_request_and_tools() -> None:
    tc = SimpleNamespace(id="c1", function=SimpleNamespace(name="n", arguments='{"a": 2}'))
    resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[tc]), finish_reason="tool_calls"
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=3,
            prompt_tokens_details=SimpleNamespace(cached_tokens=8),
        ),
        model="openai/gpt-oss-120b:free",
    )
    stub = StubOpenAI(resp)
    out = OpenRouterAdapter(client=stub).raw_complete(
        model="openai/gpt-oss-120b:free",
        messages=CONVO,
        tools=TOOLS,
        json_schema={"type": "object"},
        temperature=0,
        max_tokens=50,
    )
    assert stub.kwargs["response_format"]["type"] == "json_schema"
    assert stub.kwargs["tools"][0]["type"] == "function"
    assert out.tool_calls[0].arguments == {"a": 2}
    assert out.usage.cached_input_tokens == 8
    converted = to_openai_messages(CONVO)
    assert converted[2]["tool_calls"][0]["function"]["name"] == "get_endpoint_schema"
    assert converted[3]["role"] == "tool"


class StubGemini:
    def __init__(self, resp: Any) -> None:
        self.resp = resp
        self.kwargs: dict[str, Any] = {}
        self.models = self

    def generate_content(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.resp


def test_gemini_request_and_usage() -> None:
    from google.genai import types

    content = types.Content(role="model", parts=[types.Part(text='{"x": 1}')])
    resp = types.GenerateContentResponse(
        candidates=[types.Candidate(content=content, finish_reason=types.FinishReason.STOP)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=50,
            candidates_token_count=7,
            cached_content_token_count=30,
            thoughts_token_count=3,
        ),
        model_version="gemini-3.8-flash",
    )
    stub = StubGemini(resp)
    out = GeminiAdapter(client=stub).raw_complete(
        model="gemini-3.8-flash",
        messages=CONVO,
        tools=TOOLS,
        json_schema={"type": "object"},
        temperature=0,
        max_tokens=50,
    )
    cfg = stub.kwargs["config"]
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema == {"type": "object"}
    assert cfg.system_instruction == "S"
    assert cfg.tools[0].function_declarations[0].name == "get_endpoint_schema"
    roles = [c.role for c in stub.kwargs["contents"]]
    assert roles == ["user", "model", "user"]
    assert out.text == '{"x": 1}'
    assert out.usage.input_tokens == 50 and out.usage.cached_input_tokens == 30
    assert out.usage.output_tokens == 10
