from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from agentqa import config
from agentqa.config import ModelRef
from agentqa.guards import killswitch
from agentqa.guards.budget import BudgetGuard
from agentqa.llm.adapters.fake import FakeAdapter
from agentqa.llm.cache import DiskCache
from agentqa.llm.pricing import CostLedger, costs, list_cost
from agentqa.llm.prompts import parse_prompt
from agentqa.llm.ratelimit import RateLimiter
from agentqa.llm.resilience import CircuitBreaker, backoff_delay
from agentqa.llm.router import LLMClient, ProviderRegistry
from agentqa.llm.types import (
    AllProvidersFailed,
    BudgetExceeded,
    CacheMiss,
    KillSwitchEngaged,
    Message,
    ProviderUnavailable,
    RateLimited,
    RawResponse,
    SchemaValidationFailed,
    Usage,
)
from agentqa.obs import metrics


class Answer(BaseModel):
    value: int


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def make_client(
    tmp_path: Path,
    adapters: dict[str, FakeAdapter],
    chain: list[ModelRef],
    mode: str = "read_write",
    **kw: object,
) -> tuple[LLMClient, Clock]:
    clock = Clock()
    reg = ProviderRegistry(adapters, sleep=clock.sleep, clock=clock)  # type: ignore[arg-type]
    cache = DiskCache(tmp_path / "c", mode=mode)  # type: ignore[arg-type]
    return LLMClient(chain, registry=reg, cache=cache, **kw), clock  # type: ignore[arg-type]


MSGS = [Message(role="system", content="sys"), Message(role="user", content="what is 2+2?")]
SIM_A = ModelRef(provider="simulated", model="sim-cheap")
SIM_B = ModelRef(provider="simulated", model="sim-strong")


def test_config_profiles_load_and_judge_family_rule() -> None:
    m = config.models()
    assert {"free", "mixed", "premium", "simulated"} <= set(m.profiles)
    bad = m.model_dump()
    bad["profiles"]["free"]["judge"] = {"provider": "gemini", "model": "gemini-3.8-flash"}
    with pytest.raises(ValueError, match="self-preference"):
        config.ModelsFile.model_validate(bad)


def test_every_profile_model_is_priced() -> None:
    prices = config.pricing().models
    for prof in config.models().profiles.values():
        for tier in prof.tiers.values():
            for ref in tier.chain():
                assert ref.model in prices
        assert prof.judge.model in prices


def test_pricing_actual_vs_list() -> None:
    u = Usage(input_tokens=1_000_000, output_tokens=1_000_000, cached_input_tokens=0)
    assert list_cost("claude-opus-5-5", u) == pytest.approx(24.0)
    actual, list_eq = costs("gemini", "gemini-3.8-flash", u)
    assert actual == 0.0 and list_eq == pytest.approx(4.5)
    cached = Usage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=1_000_000)
    assert list_cost("claude-opus-5-5", cached) == pytest.approx(0.20)


def test_rate_limiter_waits_then_proceeds() -> None:
    clock = Clock()
    rl = RateLimiter(rpm=2, tpm=1000, rpd=10, clock=clock, sleep=clock.sleep)
    assert rl.acquire(10) == 0
    assert rl.acquire(10) == 0
    waited = rl.acquire(10)
    assert waited == pytest.approx(30.0, rel=0.01)


def test_rate_limiter_daily_quota() -> None:
    clock = Clock()
    rl = RateLimiter(rpm=100, tpm=10_000, rpd=2, clock=clock, sleep=clock.sleep)
    rl.acquire(1)
    rl.acquire(1)
    with pytest.raises(RateLimited):
        rl.acquire(1)


def test_backoff_and_breaker() -> None:
    import random

    d = [backoff_delay(i, 1.0, 8.0, 0.0, random.Random(1)) for i in range(1, 6)]
    assert d == [1.0, 2.0, 4.0, 8.0, 8.0]
    clock = Clock()
    cb = CircuitBreaker(failure_threshold=2, reset_after_s=10, clock=clock)
    cb.record_failure()
    assert cb.allow()
    cb.record_failure()
    assert cb.state == "open" and not cb.allow()
    clock.t += 11
    assert cb.state == "half_open" and cb.allow()
    cb.record_success()
    assert cb.state == "closed"


def test_structured_output_repair_counts_retries(tmp_path: Path) -> None:
    fake = FakeAdapter(script=[RawResponse(text="not json"), RawResponse(text='{"value": 4}')])
    client, _ = make_client(tmp_path, {"simulated": fake}, [SIM_A])
    result = client.complete(MSGS, response_schema=Answer)
    assert result.parsed == Answer(value=4)
    assert result.retry_count == 1
    # the validation error is fed back to the model
    second = fake.calls[1]["messages"]
    assert "did not validate" in second[-1].content


def test_structured_output_gives_up(tmp_path: Path) -> None:
    fake = FakeAdapter(script=[RawResponse(text="x")] * 3)
    client, _ = make_client(tmp_path, {"simulated": fake}, [SIM_A])
    with pytest.raises(SchemaValidationFailed) as exc:
        client.complete(MSGS, response_schema=Answer)
    assert exc.value.retries == 2


def test_backoff_on_429_then_success(tmp_path: Path) -> None:
    fake = FakeAdapter(
        provider="gemini", script=[RateLimited("429"), RawResponse(text='{"value": 1}')]
    )
    chain = [ModelRef(provider="gemini", model="gemini-3.8-flash")]
    client, clock = make_client(tmp_path, {"gemini": fake}, chain)
    result = client.complete(MSGS, response_schema=Answer)
    assert result.transport_retries == 1
    assert clock.slept and clock.slept[0] > 0


def test_fallback_chain_records_fallback(tmp_path: Path) -> None:
    primary = FakeAdapter(provider="gemini", script=[ProviderUnavailable("down")] * 10)
    secondary = FakeAdapter(provider="simulated", script=[RawResponse(text="ok")])
    chain = [ModelRef(provider="gemini", model="gemini-3.8-flash"), SIM_B]
    client, _ = make_client(tmp_path, {"gemini": primary, "simulated": secondary}, chain)
    before = metrics.total("agentqa_fallbacks_total")
    result = client.complete(MSGS)
    assert result.text == "ok" and result.model == "sim-strong"
    assert result.fallback_from == "gemini/gemini-3.8-flash"
    assert metrics.total("agentqa_fallbacks_total") == before + 1


def test_all_providers_failed(tmp_path: Path) -> None:
    fake = FakeAdapter(script=[ProviderUnavailable("down")] * 10)
    client, _ = make_client(tmp_path, {"simulated": fake}, [SIM_A])
    with pytest.raises(AllProvidersFailed):
        client.complete(MSGS)


def test_cache_read_write_and_replay(tmp_path: Path) -> None:
    fake = FakeAdapter(script=[RawResponse(text='{"value": 7}')])
    client, _ = make_client(tmp_path, {"simulated": fake}, [SIM_A])
    first = client.complete(MSGS, response_schema=Answer)
    assert first.cache_status == "miss" and first.cost_usd_list_equivalent > 0
    second = client.complete(MSGS, response_schema=Answer)
    assert second.cache_status == "hit" and second.parsed == Answer(value=7)
    assert second.cost_usd_list_equivalent == 0.0
    assert len(fake.calls) == 1
    replay, _ = make_client(tmp_path, {"simulated": FakeAdapter()}, [SIM_A], mode="replay_only")
    assert replay.complete(MSGS, response_schema=Answer).parsed == Answer(value=7)
    with pytest.raises(CacheMiss):
        replay.complete([*MSGS, Message(role="user", content="new")])


def test_cache_key_includes_prompt_version(tmp_path: Path) -> None:
    from agentqa.llm.types import CallMetadata

    fake = FakeAdapter(script=[RawResponse(text="a"), RawResponse(text="b")])
    client, _ = make_client(tmp_path, {"simulated": fake}, [SIM_A])
    a = client.complete(MSGS, metadata=CallMetadata(prompt_name="p", prompt_version="1.0.0"))
    b = client.complete(MSGS, metadata=CallMetadata(prompt_name="p", prompt_version="1.1.0"))
    assert (a.text, b.text) == ("a", "b")


def test_budget_guard_blocks_call(tmp_path: Path) -> None:
    ledger = CostLedger()
    guard = BudgetGuard(ledger=ledger, max_tokens=50, max_usd=1, max_wall_clock_s=100, max_steps=10)
    fake = FakeAdapter(script=[RawResponse(text="x")])
    client, _ = make_client(
        tmp_path, {"simulated": fake}, [SIM_A], ledger=ledger, pre_call=[guard.pre_call]
    )
    with pytest.raises(BudgetExceeded):
        client.complete(MSGS, max_tokens=100)
    assert fake.calls == []


def test_killswitch_blocks_call(tmp_path: Path) -> None:
    fake = FakeAdapter(script=[RawResponse(text="x")])
    client, _ = make_client(tmp_path, {"simulated": fake}, [SIM_A], pre_call=[killswitch.llm_hook])
    killswitch.engage("test")
    try:
        with pytest.raises(KillSwitchEngaged):
            client.complete(MSGS)
    finally:
        killswitch.release()
    assert client.complete(MSGS).text == "x"


def test_ledger_summary(tmp_path: Path) -> None:
    fake = FakeAdapter(
        script=[RawResponse(text="x", usage=Usage(input_tokens=100, output_tokens=10))]
    )
    client, _ = make_client(tmp_path, {"simulated": fake}, [SIM_A], tier="T1")
    client.complete(MSGS)
    s = client.ledger.summary()
    assert s["calls"] == 1 and s["total_tokens"] == 110
    assert "T1" in s["by_tier"]  # type: ignore[operator]


def test_prompt_parse_and_render() -> None:
    p = parse_prompt(
        "---\nname: t\nversion: 1.2.3\nrole: planner\n---\n## system\nS\n## user\nHi {{ x }}\n"
    )
    msgs = p.render(x="there")
    assert msgs[0].content == "S" and msgs[0].cache_prefix
    assert msgs[1].content == "Hi there"
    with pytest.raises(ValueError):
        parse_prompt("---\nname: t\nversion: 1.2\nrole: r\n---\n## system\nS\n## user\nU")


def test_llm_span_attributes(tmp_path: Path) -> None:
    from agentqa.llm.types import CallMetadata
    from agentqa.obs import tracing

    fake = FakeAdapter(script=[RawResponse(text="x", usage=Usage(input_tokens=5, output_tokens=2))])
    client, _ = make_client(tmp_path, {"simulated": fake}, [SIM_A])
    with tracing.span("run test", "run") as root:
        client.complete(
            MSGS,
            metadata=CallMetadata(agent="planner", prompt_name="planner", prompt_version="1.0.0"),
        )
        tid = format(root.get_span_context().trace_id, "032x")
    spans = tracing.finished_spans(tid)
    llm = next(s for s in spans if s["attributes"].get("agentqa.span_kind") == "llm_call")
    a = llm["attributes"]
    assert a["gen_ai.request.model"] == "sim-cheap"
    assert a["gen_ai.usage.input_tokens"] == 5
    assert a["agentqa.prompt_version"] == "1.0.0"
    assert a["agentqa.cache_status"] == "miss"
    assert "agentqa.cost_usd_list_equivalent" in a
    json.dumps(spans, default=str)
