"""Per-provider token-bucket rate limiter (requests/min, tokens/min, requests/day).

Thread-safe; clock and sleep are injectable so tests run instantly.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from agentqa.llm.types import RateLimited


@dataclass
class TokenBucket:
    capacity: float
    refill_per_s: float
    tokens: float = field(init=False)
    updated: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.updated = 0.0

    def _refill(self, now: float) -> None:
        if self.updated == 0.0:
            self.updated = now
        elapsed = max(0.0, now - self.updated)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_s)
        self.updated = now

    def wait_time(self, amount: float, now: float) -> float:
        self._refill(now)
        amount = min(amount, self.capacity)  # a single oversized request still proceeds eventually
        if self.tokens >= amount:
            return 0.0
        return (amount - self.tokens) / self.refill_per_s

    def take(self, amount: float, now: float) -> None:
        self._refill(now)
        self.tokens -= min(amount, self.capacity)


class RateLimiter:
    def __init__(
        self,
        rpm: int,
        tpm: int,
        rpd: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        max_wait_s: float = 120.0,
    ) -> None:
        self.requests = TokenBucket(capacity=rpm, refill_per_s=rpm / 60.0)
        self.tokens = TokenBucket(capacity=tpm, refill_per_s=tpm / 60.0)
        self.rpd = rpd
        self.day_count = 0
        self.day_started = clock()
        self.clock = clock
        self.sleep = sleep
        self.max_wait_s = max_wait_s
        self.total_waited_s = 0.0
        self._lock = threading.Lock()

    def acquire(self, est_tokens: int) -> float:
        """Block until one request of ``est_tokens`` fits. Returns seconds waited.

        Raises RateLimited when the daily quota is exhausted or the wait would exceed
        ``max_wait_s`` (the router then falls back to the next provider).
        """
        waited = 0.0
        while True:
            with self._lock:
                now = self.clock()
                if now - self.day_started >= 86400:
                    self.day_started, self.day_count = now, 0
                if self.day_count >= self.rpd:
                    raise RateLimited("daily request quota exhausted")
                wait = max(self.requests.wait_time(1, now), self.tokens.wait_time(est_tokens, now))
                if wait <= 0:
                    self.requests.take(1, now)
                    self.tokens.take(est_tokens, now)
                    self.day_count += 1
                    self.total_waited_s += waited
                    return waited
            if waited + wait > self.max_wait_s:
                raise RateLimited(f"local rate limit wait {wait:.1f}s exceeds max")
            self.sleep(wait)
            waited += wait
