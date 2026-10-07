"""LLMClient: one call path for every agent.

``complete()`` walks the fallback chain for a role/tier. For each provider/model it:
checks the kill switch and budget, consults the disk cache, acquires the rate limiter,
calls the adapter with exponential backoff on 429/5xx, trips the circuit breaker on repeated
failure, validates structured output with pydantic (feeding the validation error back to the
model for up to N repair retries), prices the call, writes the cost ledger, and records an
``llm_call`` span with GenAI semantic-convention and ``agentqa.*`` attributes.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from agentqa import config
from agentqa.config import ModelRef, Profile
from agentqa.llm.adapters.base import ProviderAdapter, estimate_tokens, messages_text, strict_schema
from agentqa.llm.cache import DiskCache, cache_key
from agentqa.llm.pricing import CostLedger, LedgerEntry, costs
from agentqa.llm.ratelimit import RateLimiter
from agentqa.llm.resilience import CircuitBreaker, backoff_delay
from agentqa.llm.types import (
    AllProvidersFailed,
    CacheMiss,
    CallMetadata,
    LLMError,
    LLMResult,
    Message,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
    RawResponse,
    SchemaValidationFailed,
    ToolSpec,
    Usage,
)
from agentqa.obs import metrics, tracing
from agentqa.obs.logging import get_logger

log = get_logger("agentqa.llm")

# Hooks the guards install (budget guard, kill switch) without the LLM layer importing them.
PreCallHook = Callable[[CallMetadata, int], None]
PostCallHook = Callable[[CallMetadata, LLMResult], None]


def _extract_json(text: str) -> str:
    """Accept a bare JSON document or one wrapped in a Markdown code fence."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


class ProviderRegistry:
    """Shared, process-wide adapters, rate limiters and circuit breakers."""

    def __init__(
        self,
        adapters: dict[str, ProviderAdapter] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._adapters: dict[str, ProviderAdapter] = dict(adapters or {})
        self._limiters: dict[str, RateLimiter] = {}
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
        self.sleep = sleep
        self.clock = clock

    def adapter(self, provider: str) -> ProviderAdapter:
        with self._lock:
            if provider not in self._adapters:
                self._adapters[provider] = _build_adapter(provider)
            return self._adapters[provider]

    def limiter(self, provider: str) -> RateLimiter:
        with self._lock:
            if provider not in self._limiters:
                pc = config.providers().providers[provider]
                self._limiters[provider] = RateLimiter(
                    pc.rpm, pc.tpm, pc.rpd, clock=self.clock, sleep=self.sleep
                )
            return self._limiters[provider]

    def breaker(self, provider: str, model: str) -> CircuitBreaker:
        key = f"{provider}/{model}"
        with self._lock:
            if key not in self._breakers:
                cb = config.providers().circuit_breaker
                self._breakers[key] = CircuitBreaker(
                    cb.failure_threshold, cb.reset_after_s, clock=self.clock
                )
            return self._breakers[key]


def _build_adapter(provider: str) -> ProviderAdapter:
    if provider == "anthropic":
        from agentqa.llm.adapters.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter()
    if provider == "gemini":
        from agentqa.llm.adapters.gemini_adapter import GeminiAdapter

        return GeminiAdapter()
    if provider == "openrouter":
        from agentqa.llm.adapters.openrouter_adapter import OpenRouterAdapter

        return OpenRouterAdapter()
    if provider == "simulated":
        from agentqa.llm.simulated import SimulatedAdapter

        return SimulatedAdapter()
    raise KeyError(f"unknown provider {provider!r}")


class LLMClient:
    """A client bound to an ordered fallback chain of (provider, model)."""

    def __init__(
        self,
        chain: list[ModelRef],
        *,
        registry: ProviderRegistry,
        cache: DiskCache,
        ledger: CostLedger | None = None,
        tier: str | None = None,
        pre_call: list[PreCallHook] | None = None,
        post_call: list[PostCallHook] | None = None,
        rng: random.Random | None = None,
        native_cache: bool = True,
    ) -> None:
        if not chain:
            raise ValueError("empty model chain")
        self.chain = chain
        self.registry = registry
        self.cache = cache
        self.ledger = ledger if ledger is not None else CostLedger()
        self.tier = tier
        self.pre_call = pre_call or []
        self.post_call = post_call or []
        self.rng = rng or random.Random(0)
        self.native_cache = native_cache

    @property
    def primary(self) -> ModelRef:
        return self.chain[0]

    # ------------------------------------------------------------------ public API

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        metadata: CallMetadata | None = None,
    ) -> LLMResult:
        meta = metadata or CallMetadata()
        if not self.native_cache:
            messages = [m.model_copy(update={"cache_prefix": False}) for m in messages]
        if meta.tier is None and self.tier is not None:
            meta = meta.model_copy(update={"tier": self.tier})
        est = estimate_tokens(messages_text(messages)) + max_tokens
        for hook in self.pre_call:
            hook(meta, est)

        json_schema = (
            strict_schema(response_schema.model_json_schema()) if response_schema else None
        )
        errors: list[str] = []
        fallback_from: str | None = None
        for idx, ref in enumerate(self.chain):
            breaker = self.registry.breaker(ref.provider, ref.model)
            if not breaker.allow():
                errors.append(f"{ref.provider}/{ref.model}: circuit open")
                tracing.event("circuit_open", provider=ref.provider, model=ref.model)
                fallback_from = fallback_from or f"{ref.provider}/{ref.model}"
                continue
            try:
                result = self._complete_with(
                    ref,
                    messages,
                    tools,
                    response_schema,
                    json_schema,
                    temperature,
                    max_tokens,
                    meta,
                )
            except CacheMiss:
                raise
            except SchemaValidationFailed:
                raise
            except LLMError as exc:
                breaker.record_failure()
                errors.append(f"{ref.provider}/{ref.model}: {type(exc).__name__}: {exc}")
                metrics.inc(
                    "agentqa_llm_errors_total", provider=ref.provider, type=type(exc).__name__
                )
                nxt = self.chain[idx + 1] if idx + 1 < len(self.chain) else None
                if nxt is not None:
                    labels = {
                        "from": f"{ref.provider}/{ref.model}",
                        "to": f"{nxt.provider}/{nxt.model}",
                    }
                    metrics.inc("agentqa_fallbacks_total", 1, **labels)
                    tracing.event(
                        "fallback",
                        from_model=f"{ref.provider}/{ref.model}",
                        to_model=f"{nxt.provider}/{nxt.model}",
                        reason=type(exc).__name__,
                    )
                    log.warning(
                        "llm_fallback",
                        provider=ref.provider,
                        model=ref.model,
                        error=type(exc).__name__,
                    )
                fallback_from = fallback_from or f"{ref.provider}/{ref.model}"
                continue
            breaker.record_success()
            if fallback_from:
                result.fallback_from = fallback_from
            for post in self.post_call:
                post(meta, result)
            return result
        raise AllProvidersFailed("; ".join(errors))

    # ------------------------------------------------------------------ internals

    def _complete_with(
        self,
        ref: ModelRef,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        response_schema: type[BaseModel] | None,
        json_schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
        meta: CallMetadata,
    ) -> LLMResult:
        max_repairs = config.providers().structured_output.max_repair_retries
        convo = list(messages)
        total_usage = Usage()
        transport_retries = 0
        cache_statuses: list[str] = []
        started = time.perf_counter()
        last_text = ""
        with tracing.span(
            f"llm {meta.agent}",
            "llm_call",
            **{
                tracing.GEN_AI_OPERATION: "chat",
                tracing.GEN_AI_PROVIDER: ref.provider,
                tracing.GEN_AI_REQUEST_MODEL: ref.model,
                "llm.model_name": ref.model,
                "llm.provider": ref.provider,
                "agentqa.agent": meta.agent,
                "agentqa.prompt_name": meta.prompt_name,
                "agentqa.prompt_version": meta.prompt_version,
                "agentqa.prompt_hash": meta.prompt_hash,
                "agentqa.task_id": meta.task_id,
                "agentqa.tier": meta.tier,
                "input.value": messages[-1].content[:2000] if messages else "",
            },
        ) as sp:
            for attempt in range(max_repairs + 1):
                raw, status, retries = self._one_call(
                    ref, convo, tools, json_schema, temperature, max_tokens, meta
                )
                transport_retries += retries
                cache_statuses.append(status)
                total_usage = total_usage + raw.usage
                last_text = raw.text
                if response_schema is None or raw.tool_calls:
                    parsed = None
                    break
                try:
                    parsed = response_schema.model_validate_json(_extract_json(raw.text))
                    break
                except (ValidationError, ValueError) as exc:
                    err = str(exc)[:1500]
                    tracing.event("schema_validation_failed", attempt=attempt, error=err[:300])
                    metrics.inc(
                        "agentqa_guardrail_events_total", guardrail="schema", action="repair"
                    )
                    if attempt >= max_repairs:
                        metrics.inc(
                            "agentqa_guardrail_events_total", guardrail="schema", action="reject"
                        )
                        self._record(
                            ref,
                            meta,
                            total_usage,
                            cache_statuses,
                            sp,
                            attempt,
                            transport_retries,
                            time.perf_counter() - started,
                            raw,
                        )
                        raise SchemaValidationFailed(
                            err, last_text=last_text, retries=attempt
                        ) from exc
                    convo = [
                        *convo,
                        Message(role="assistant", content=raw.text),
                        Message(
                            role="user",
                            content=(
                                "Your previous reply did not validate against the required JSON schema.\n"
                                f"Validation error:\n{err}\n"
                                "Reply again with only a JSON document that satisfies the schema."
                            ),
                        ),
                    ]
            latency = time.perf_counter() - started
            result = self._record(
                ref, meta, total_usage, cache_statuses, sp, attempt, transport_retries, latency, raw
            )
            result.parsed = parsed
            result.tool_calls = raw.tool_calls
            sp.set_attribute("output.value", raw.text[:2000])
            return result

    def _one_call(
        self,
        ref: ModelRef,
        convo: list[Message],
        tools: list[ToolSpec] | None,
        json_schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
        meta: CallMetadata,
    ) -> tuple[RawResponse, str, int]:
        key = cache_key(
            {
                "provider": ref.provider,
                "model": ref.model,
                "messages": [m.model_dump(exclude={"provider_raw"}) for m in convo],
                "tools": [t.model_dump() for t in tools or []],
                "schema": json_schema,
                "params": {"temperature": temperature, "max_tokens": max_tokens},
                "prompt": [meta.prompt_name, meta.prompt_version, meta.prompt_hash],
            }
        )
        cached = self.cache.get(key)  # raises CacheMiss in replay_only mode
        if cached is not None:
            metrics.inc("agentqa_cache_hits_total", agent=meta.agent)
            return cached, "hit", 0
        pc = config.providers().providers[ref.provider]
        bo = config.providers().backoff
        adapter = self.registry.adapter(ref.provider)
        limiter = self.registry.limiter(ref.provider)
        est = estimate_tokens(messages_text(convo)) + max_tokens
        attempt = 0
        while True:
            limiter.acquire(est)
            try:
                raw = adapter.raw_complete(
                    model=ref.model,
                    messages=convo,
                    tools=tools,
                    json_schema=json_schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                break
            except ProviderError:
                raise
            except (RateLimited, ProviderUnavailable) as exc:
                attempt += 1
                metrics.inc(
                    "agentqa_llm_errors_total", provider=ref.provider, type=type(exc).__name__
                )
                if attempt > pc.max_retries:
                    raise
                delay = backoff_delay(attempt, bo.base_s, bo.max_s, bo.jitter, self.rng)
                if isinstance(exc, RateLimited) and exc.retry_after_s:
                    delay = max(delay, exc.retry_after_s)
                tracing.event(
                    "retry", attempt=attempt, delay_s=round(delay, 2), error=type(exc).__name__
                )
                self.registry.sleep(delay)
        status = "off" if self.cache.mode == "off" else "miss"
        self.cache.put(key, raw)
        return raw, status, attempt

    def _record(
        self,
        ref: ModelRef,
        meta: CallMetadata,
        usage: Usage,
        cache_statuses: list[str],
        sp: Any,
        repairs: int,
        transport_retries: int,
        latency: float,
        raw: RawResponse,
    ) -> LLMResult:
        all_hits = all(s == "hit" for s in cache_statuses)
        # A disk-cache replay costs nothing; the original cost is not re-charged.
        actual, list_eq = (0.0, 0.0) if all_hits else costs(ref.provider, ref.model, usage)
        cache_status = (
            "hit" if all_hits else ("off" if all(s == "off" for s in cache_statuses) else "miss")
        )
        sp.set_attribute(tracing.GEN_AI_RESPONSE_MODEL, raw.response_model or ref.model)
        sp.set_attribute(tracing.GEN_AI_INPUT_TOKENS, usage.input_tokens)
        sp.set_attribute(tracing.GEN_AI_OUTPUT_TOKENS, usage.output_tokens)
        sp.set_attribute(tracing.GEN_AI_CACHE_READ_TOKENS, usage.cached_input_tokens)
        sp.set_attribute(tracing.GEN_AI_FINISH_REASONS, [raw.finish_reason])
        sp.set_attribute("llm.token_count.prompt", usage.input_tokens)
        sp.set_attribute("llm.token_count.completion", usage.output_tokens)
        sp.set_attribute("llm.token_count.total", usage.total)
        sp.set_attribute("agentqa.cache_status", cache_status)
        sp.set_attribute("agentqa.retry_count", repairs)
        sp.set_attribute("agentqa.transport_retries", transport_retries)
        sp.set_attribute("agentqa.cost_usd_actual", actual)
        sp.set_attribute("agentqa.cost_usd_list_equivalent", list_eq)
        sp.set_attribute("agentqa.cached_input_tokens", usage.cached_input_tokens)
        sp.set_attribute("agentqa.latency_s", round(latency, 4))
        if not all_hits:
            labels = {"agent": meta.agent, "provider": ref.provider, "model": ref.model}
            metrics.inc("agentqa_llm_tokens_total", usage.input_tokens, direction="input", **labels)
            metrics.inc(
                "agentqa_llm_tokens_total", usage.output_tokens, direction="output", **labels
            )
            metrics.inc(
                "agentqa_llm_tokens_total",
                usage.cached_input_tokens,
                direction="cached_input",
                **labels,
            )
            metrics.inc("agentqa_llm_cost_usd_total", actual, kind="actual")
            metrics.inc("agentqa_llm_cost_usd_total", list_eq, kind="list_equivalent")
            metrics.observe("agentqa_llm_latency_seconds", latency, **labels)
        self.ledger.add(
            LedgerEntry(
                agent=meta.agent,
                provider=ref.provider,
                model=ref.model,
                tier=meta.tier,
                usage=Usage() if all_hits else usage,
                cost_usd_actual=actual,
                cost_usd_list_equivalent=list_eq,
                cache_status=cache_status,
                task_id=meta.task_id,
            )
        )
        assistant = Message(
            role="assistant",
            content=raw.text,
            tool_calls=raw.tool_calls,
            provider_raw=raw.provider_raw,
        )
        return LLMResult(
            text=raw.text,
            tool_calls=raw.tool_calls,
            usage=usage,
            provider=ref.provider,
            model=ref.model,
            response_model=raw.response_model,
            finish_reason=raw.finish_reason,
            latency_s=latency,
            cache_status=cache_status,  # type: ignore[arg-type]
            retry_count=repairs,
            transport_retries=transport_retries,
            cost_usd_actual=actual,
            cost_usd_list_equivalent=list_eq,
            prompt_name=meta.prompt_name,
            prompt_version=meta.prompt_version,
            assistant_message=assistant,
        )


class ModelRouter:
    """Builds LLMClients for roles and tiers of a profile, sharing cache, ledger and hooks."""

    def __init__(
        self,
        profile_name: str | None = None,
        *,
        registry: ProviderRegistry | None = None,
        cache: DiskCache | None = None,
        ledger: CostLedger | None = None,
        pre_call: list[PreCallHook] | None = None,
        post_call: list[PostCallHook] | None = None,
        native_cache: bool = True,
    ) -> None:
        self.native_cache = native_cache
        self.profile_name = profile_name or os.environ.get("AGENTQA_PROFILE", "simulated")
        self.profile: Profile = config.profile(self.profile_name)
        self.registry = registry or ProviderRegistry()
        self.cache = cache or DiskCache()
        self.ledger = ledger if ledger is not None else CostLedger()
        self.pre_call = pre_call or []
        self.post_call = post_call or []

    def for_tier(self, tier: str) -> LLMClient:
        tc = self.profile.tiers[tier]
        return LLMClient(
            tc.chain(),
            registry=self.registry,
            cache=self.cache,
            ledger=self.ledger,
            tier=tier,
            pre_call=self.pre_call,
            post_call=self.post_call,
            native_cache=self.native_cache,
        )

    def for_role(self, role: str) -> LLMClient:
        if role == "judge":
            return LLMClient(
                [self.profile.judge],
                registry=self.registry,
                cache=self.cache,
                ledger=self.ledger,
                tier="judge",
                pre_call=self.pre_call,
                post_call=self.post_call,
                native_cache=self.native_cache,
            )
        return self.for_tier(self.profile.tier_for_role(role))


def tool_result_message(call_id: str, name: str, payload: Any) -> Message:
    content = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return Message(role="tool", content=content[:6000], tool_call_id=call_id, name=name)
