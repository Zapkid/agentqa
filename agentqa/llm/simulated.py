"""Deterministic simulated models (no network).

Why it exists: the build and CI have no API keys, yet the dispatcher, cascade, guards and eval
harness must be exercised end to end. The simulator stands in for a "cheap" and a "strong" model
with *declared* error rates, so verification-gated escalation has real failures to catch.
Numbers produced with it measure the pipeline's mechanics, never a real model's quality, and
every report built on it says so.

Each prompt registers a responder: ``fn(request) -> RawResponse``. The adapter identifies the
prompt by its system text (the system section of a registered prompt file). Randomness comes
from a seed derived from the request content and the model, so identical requests give
identical answers.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentqa.llm.adapters.base import estimate_tokens, messages_text
from agentqa.llm.prompts import all_prompts
from agentqa.llm.types import Message, ProviderError, RawResponse, ToolCall, ToolSpec, Usage

# Declared error rates: the probability that a simulated model makes a mistake on a task of
# difficulty 1.0 (scaled linearly by the task's difficulty).
SKILL = {
    "sim-cheap": {"error_rate": 0.45, "schema_error_rate": 0.08},
    "sim-strong": {"error_rate": 0.06, "schema_error_rate": 0.01},
    "sim-judge": {"error_rate": 0.10, "schema_error_rate": 0.01},
}


@dataclass
class SimRequest:
    model: str
    messages: list[Message]
    tools: list[ToolSpec] | None
    json_schema: dict[str, Any] | None
    rng: random.Random
    error_rate: float
    schema_error_rate: float

    @property
    def user(self) -> str:
        users = [m.content for m in self.messages if m.role == "user"]
        return users[0] if users else ""

    @property
    def last(self) -> Message:
        return self.messages[-1]

    def payload(self) -> dict[str, Any]:
        """Responders receive their structured input as a JSON block between
        ``<input>`` tags in the user message (prompts render it that way)."""
        text = self.user
        if "<input>" in text and "</input>" in text:
            blob = text.split("<input>", 1)[1].split("</input>", 1)[0]
            try:
                data = json.loads(blob)
                return data if isinstance(data, dict) else {"items": data}
            except json.JSONDecodeError:
                return {}
        return {}

    def mistake(self, difficulty: float = 0.5) -> bool:
        return self.rng.random() < self.error_rate * max(0.0, min(1.0, difficulty)) * 2

    def tool_results(self) -> dict[str, str]:
        return {m.name or "": m.content for m in self.messages if m.role == "tool"}

    def is_repair(self) -> bool:
        return len([m for m in self.messages if m.role == "user"]) > 1


Responder = Callable[[SimRequest], RawResponse | str | dict[str, Any] | list[ToolCall]]
RESPONDERS: dict[str, Responder] = {}


def responder(prompt_name: str) -> Callable[[Responder], Responder]:
    def deco(fn: Responder) -> Responder:
        RESPONDERS[prompt_name] = fn
        return fn

    return deco


def _system_index() -> dict[str, str]:
    return {p.system: p.name for p in all_prompts()}


class SimulatedAdapter:
    provider = "simulated"

    def __init__(self) -> None:
        # Responders live next to the agents; importing registers them.
        import agentqa.sim  # noqa: F401

        self._by_system = _system_index()
        self._seen_prefixes: set[str] = set()
        # Repetition seed for evals (N>=3 runs need run-to-run variance). Runs that compare
        # seeds must use a cold cache, since the seed is not part of the cache key.
        self.seed = os.environ.get("AGENTQA_SIM_SEED", "0")

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
        system = next((m.content for m in messages if m.role == "system"), "")
        prompt_name = self._by_system.get(system)
        if prompt_name is None or prompt_name not in RESPONDERS:
            raise ProviderError(f"simulated: no responder for prompt {prompt_name!r}")
        skill = SKILL.get(model)
        if skill is None:
            raise ProviderError(f"simulated: unknown model {model}")
        digest = hashlib.sha256((self.seed + model + messages_text(messages)).encode()).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        req = SimRequest(
            model,
            messages,
            tools,
            json_schema,
            rng,
            skill["error_rate"],
            skill["schema_error_rate"],
        )
        out = RESPONDERS[prompt_name](req)
        if isinstance(out, RawResponse):
            raw = out
        elif isinstance(out, list):
            raw = RawResponse(tool_calls=out, finish_reason="tool_use")
        else:
            text = out if isinstance(out, str) else json.dumps(out)
            # Schema slips (truncated JSON) on the first attempt only, so repair can succeed.
            if (
                json_schema is not None
                and not req.is_repair()
                and rng.random() < req.schema_error_rate
            ):
                text = text[: max(1, len(text) // 2)]
            raw = RawResponse(text=text)
        prefix_tokens = estimate_tokens(system)
        # Simulated provider-native prompt caching: a static prefix marked cache_prefix is billed
        # as cached input the second time the same model sees it (as Anthropic/Gemini do).
        cached = 0
        if messages and messages[0].cache_prefix:
            key = hashlib.sha256((model + system).encode()).hexdigest()
            if key in self._seen_prefixes:
                cached = prefix_tokens
            self._seen_prefixes.add(key)
        usage = Usage(
            input_tokens=estimate_tokens(messages_text(messages)),
            output_tokens=estimate_tokens(
                raw.text + json.dumps([t.arguments for t in raw.tool_calls])
            ),
            cached_input_tokens=cached,
        )
        usage.input_tokens = max(usage.input_tokens, prefix_tokens, cached)
        return raw.model_copy(update={"usage": usage, "response_model": model})
