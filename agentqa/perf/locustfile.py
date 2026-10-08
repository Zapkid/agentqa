"""Generic Locust driver for a validated WorkloadSpec (run by ``python -m locust -f`` in a subprocess).

Nothing here is model-written: the WorkloadSpec (scenarios, weights, think time) arrives as JSON
in ``AGENTQA_PERF_CONFIG`` and is mapped onto Locust tasks and a stepped LoadTestShape.

Load guard (guardrail 9), enforced inside the load process so it reacts within a second:
users are capped, the run aborts when the target's error rate over the last 200 requests exceeds
the threshold, when host CPU or memory crosses its threshold, or when the kill switch appears.
Every request is recorded (timestamp, scenario, latency, bytes, status, users) for analysis.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import time
import uuid
from typing import Any

import gevent
import psutil
from locust import HttpUser, LoadTestShape, between, constant, events

with open(os.environ["AGENTQA_PERF_CONFIG"], encoding="utf-8") as _fh:
    CFG: dict[str, Any] = json.loads(_fh.read())
RECORDS: list[list[Any]] = []
POOLS: dict[str, list[str]] = {"order_id": [], "product_id": [], "customer_id": []}
STATE: dict[str, Any] = {"aborted": None, "env": None}
EXPECTED = {s["name"]: set(s["expected_status"]) for s in CFG["scenarios"]}
GUARD = CFG["guard"]
RPS_WINDOW_S = 5.0  # request rate is averaged over this many seconds


def _users() -> int:
    env = STATE["env"]
    return int(env.runner.user_count) if env is not None and env.runner is not None else 0


@events.test_start.add_listener
def on_test_start(environment: Any, **_: Any) -> None:
    import httpx

    STATE["env"] = environment
    headers = CFG["headers"]
    with httpx.Client(base_url=environment.host, headers=headers, timeout=30) as c:
        orders = c.get("/orders", params={"page_size": 100}).json().get("items", [])
        POOLS["order_id"] = [o["id"] for o in orders] or [str(uuid.uuid4())]
        POOLS["customer_id"] = list({o["customer_id"] for o in orders}) or [str(uuid.uuid4())]
        products = c.get("/products").json()
        POOLS["product_id"] = [p["id"] for p in products if p.get("active")] or [str(uuid.uuid4())]
    gevent.spawn(_guard_loop, environment)


def _abort(environment: Any, reason: str) -> None:
    if STATE["aborted"] is None:
        STATE["aborted"] = reason
        environment.runner.quit()


def _guard_loop(environment: Any) -> None:
    psutil.cpu_percent(interval=None)
    while environment.runner is not None and STATE["aborted"] is None:
        gevent.sleep(1.0)
        if os.path.exists(GUARD["killswitch_file"]):
            _abort(environment, "kill switch engaged")
        recent = RECORDS[-200:]
        if len(recent) >= 50:
            bad = sum(1 for r in recent if r[4] >= 500 or r[4] == 0)
            if bad / len(recent) > GUARD["abort_error_rate"]:
                _abort(
                    environment,
                    f"target error rate {bad / len(recent):.0%} over the last {len(recent)} requests",
                )
        now = time.time()
        in_window = 0
        for r in reversed(RECORDS):
            if r[0] < now - RPS_WINDOW_S:
                break
            in_window += 1
        if in_window / RPS_WINDOW_S > GUARD["max_rps"]:
            _abort(
                environment,
                f"request rate {in_window / RPS_WINDOW_S:.0f}/s above the {GUARD['max_rps']}/s cap",
            )
        if psutil.cpu_percent(interval=None) > GUARD["abort_host_cpu_pct"]:
            GUARD["_cpu_hot"] = GUARD.get("_cpu_hot", 0) + 1
            if GUARD["_cpu_hot"] >= 5:  # sustained, not a blip
                _abort(environment, "host CPU above threshold for 5s")
        else:
            GUARD["_cpu_hot"] = 0
        if psutil.virtual_memory().percent > GUARD["abort_host_mem_pct"]:
            _abort(environment, "host memory above threshold")


@events.request.add_listener
def on_request(
    request_type: str,
    name: str,
    response_time: float,
    response_length: int,
    response: Any,
    exception: Any,
    **_: Any,
) -> None:
    status = int(getattr(response, "status_code", 0) or 0)
    # record: [ts, scenario, latency_ms, bytes, status, users, ok]
    RECORDS.append(
        [
            time.time(),
            name,
            round(float(response_time), 3),
            int(response_length or 0),
            status,
            _users(),
            exception is None,
        ]
    )


@events.quitting.add_listener
def on_quitting(environment: Any, **_: Any) -> None:
    with open(CFG["out"], "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"aborted": STATE["aborted"]}) + "\n")
        for r in RECORDS:
            fh.write(json.dumps(r) + "\n")


def _fill(path: str) -> str:
    for key, pool in POOLS.items():
        path = path.replace("{" + key + "}", random.choice(pool))  # noqa: S311 - think-time, not cryptography
    return path


def _body(kind: str) -> tuple[bytes | None, dict[str, str]]:
    if kind == "order":
        body = {
            "customer_id": CFG.get("customer_id"),
            "items": [{"product_id": random.choice(POOLS["product_id"]), "quantity": 1}],  # noqa: S311 - scenario choice, not cryptography
        }
        return json.dumps(body).encode(), {"Content-Type": "application/json"}
    if kind == "webhook":
        event = {
            "event_id": f"perf-{uuid.uuid4()}",
            "order_id": str(uuid.uuid4()),
            "amount": "1.00",
            "status": "succeeded",
        }
        raw = json.dumps(event).encode()
        sig = hmac.new(CFG["webhook_secret"].encode(), raw, hashlib.sha256).hexdigest()
        return raw, {"Content-Type": "application/json", CFG["webhook_header"]: sig}
    if kind == "example":
        return json.dumps(CFG.get("examples", {}).get(kind, {})).encode(), {
            "Content-Type": "application/json"
        }
    return None, {}


def _make_task(s: dict[str, Any]) -> Any:
    def run(user: HttpUser) -> None:
        body, extra = _body(s["body"])
        with user.client.request(
            s["method"],
            _fill(s["path"]),
            params=s.get("params") or None,
            data=body,
            headers={**CFG["headers"], **extra},
            name=s["name"],
            catch_response=True,
        ) as r:
            if r.status_code in EXPECTED[s["name"]]:
                r.success()
            else:
                r.failure(f"status {r.status_code}")

    run.__name__ = s["name"]
    return run


class ApiUser(HttpUser):
    think = float(CFG["think_time_s"])
    wait_time = between(think * 0.5, think * 1.5) if think > 0 else constant(0)
    tasks = {_make_task(s): int(s["weight"]) for s in CFG["scenarios"]}  # type: ignore[assignment]  # noqa: RUF012


class SteppedShape(LoadTestShape):
    """Steps: [[duration_s, users, spawn_rate], ...] executed in order, users capped by the guard."""

    steps = CFG["steps"]

    def tick(self) -> tuple[int, float] | None:
        t = self.get_run_time()
        elapsed = 0.0
        for duration, users, rate in self.steps:
            elapsed += duration
            if t < elapsed:
                return min(int(users), int(GUARD["max_users"])), float(rate)
        return None
