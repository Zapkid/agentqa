"""Exponential backoff with jitter and a per-model circuit breaker."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field


def backoff_delay(
    attempt: int, base_s: float, max_s: float, jitter: float, rng: random.Random | None = None
) -> float:
    """Delay before retry ``attempt`` (1-based): base * 2^(attempt-1), capped, +/- jitter."""
    rng = rng or random.Random()
    delay = min(max_s, base_s * (2 ** (attempt - 1)))
    return max(0.0, delay * (1 + rng.uniform(-jitter, jitter)))


@dataclass
class CircuitBreaker:
    """Closed -> open after N consecutive failures; half-open after ``reset_after_s``."""

    failure_threshold: int = 3
    reset_after_s: float = 60.0
    clock: Callable[[], float] = time.monotonic
    failures: int = 0
    opened_at: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self.clock() - self.opened_at >= self.reset_after_s:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self.opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.failure_threshold or self.state == "half_open":
                self.opened_at = self.clock()
