"""Provider-neutral request/response types for the LLM layer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None  # for role == "tool"
    name: str | None = None  # tool name for role == "tool"
    # Provider-native assistant content (e.g. Anthropic thinking blocks, Gemini thought
    # signatures) replayed verbatim when the next turn goes to the same provider.
    provider_raw: dict[str, Any] | None = None
    # Untrusted content (retrieved docs, uploads) is delimited; see guards/injection.py.
    cache_prefix: bool = False  # mark a large static prefix for provider-native caching


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class Usage(BaseModel):
    input_tokens: int = 0  # total input tokens, including cached ones
    output_tokens: int = 0
    cached_input_tokens: int = 0  # served from provider-native prompt cache
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class RawResponse(BaseModel):
    """What an adapter returns for one HTTP round-trip (this is what the disk cache stores)."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str = "stop"
    response_model: str = ""
    provider_raw: dict[str, Any] | None = None


class CallMetadata(BaseModel):
    agent: str = "unknown"
    prompt_name: str = "adhoc"
    prompt_version: str = "0.0.0"
    prompt_hash: str = ""
    run_id: str | None = None
    task_id: str | None = None
    tier: str | None = None


CacheStatus = Literal["hit", "miss", "off"]


class LLMResult(BaseModel):
    text: str
    parsed: Any | None = None  # validated pydantic object when response_schema was given
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    provider: str
    model: str
    response_model: str = ""
    finish_reason: str = "stop"
    latency_s: float = 0.0
    cache_status: CacheStatus = "off"
    retry_count: int = 0  # structured-output repair retries
    transport_retries: int = 0  # 429/5xx retries
    cost_usd_actual: float = 0.0
    cost_usd_list_equivalent: float = 0.0
    fallback_from: str | None = None
    prompt_name: str = "adhoc"
    prompt_version: str = "0.0.0"
    assistant_message: Message | None = None


# ---------------------------------------------------------------- errors


class LLMError(Exception):
    """Base class for LLM layer errors."""


class RateLimited(LLMError):
    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ProviderUnavailable(LLMError):
    """5xx, timeouts, connection errors, refusals: retry, then fall back."""


class ProviderError(LLMError):
    """Non-retryable request error (400, auth). Falls back without retrying."""


class CacheMiss(LLMError):
    """Raised in replay_only mode when a request is not in the cache."""


class SchemaValidationFailed(LLMError):
    def __init__(self, message: str, last_text: str, retries: int) -> None:
        super().__init__(message)
        self.last_text = last_text
        self.retries = retries


class AllProvidersFailed(LLMError):
    pass


class BudgetExceeded(LLMError):
    pass


class KillSwitchEngaged(LLMError):
    pass
