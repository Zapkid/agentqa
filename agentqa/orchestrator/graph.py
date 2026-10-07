"""Supervisor: a typed task DAG executed with priorities, fan-out/fan-in and checkpoints.

- A task runs when all its dependencies are terminal; fan-in tasks set ``tolerate_failures``
  so one failed branch never blocks the rest (failure isolation).
- Ready tasks run highest priority first (risk-ordered queue) on a bounded thread pool.
- ``spawn`` derives child tasks from a task's result (planning fans out into generation).
  It runs for fresh *and* resumed results, so resume rebuilds the same graph.
- Every finished task is checkpointed to SQLite (result serialised by ``dump``); on resume,
  done tasks are loaded with ``load`` instead of re-run.
- The kill switch and the budget guard are checked before each task starts. On abort, only
  tasks marked ``always_run`` (the report) still run, so a partial report is always produced.
"""

from __future__ import annotations

import contextvars
import heapq
import itertools
import threading
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

from agentqa.llm.types import BudgetExceeded, KillSwitchEngaged
from agentqa.obs import tracing
from agentqa.obs.logging import get_logger
from agentqa.store import Store

log = get_logger("agentqa.supervisor")


@dataclass
class Task:
    id: str
    kind: str
    fn: Callable[[], Any]
    deps: list[str] = field(default_factory=list)
    priority: int = 0
    tolerate_failures: bool = False
    always_run: bool = False
    resumable: bool = True  # False: result lives only in memory, so always re-run on resume
    spawn: Callable[[Any], list[Task]] | None = None
    dump: Callable[[Any], Any] = lambda r: r
    load: Callable[[Any], Any] = lambda p: p
    status: str = "pending"  # pending | running | done | failed | skipped
    result: Any = None
    error: str | None = None


class TaskGraph:
    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def add(self, task: Task) -> Task:
        with self._lock:
            if task.id in self.tasks:
                raise ValueError(f"duplicate task id {task.id}")
            self.tasks[task.id] = task
        return task

    def result(self, task_id: str, default: Any = None) -> Any:
        t = self.tasks.get(task_id)
        return t.result if t is not None and t.status == "done" else default

    def of_kind(self, kind: str) -> list[Task]:
        return [t for t in self.tasks.values() if t.kind == kind]

    def ready(self) -> list[Task]:
        out = []
        for t in self.tasks.values():
            if t.status != "pending":
                continue
            deps = [self.tasks.get(d) for d in t.deps]
            if any(d is None or d.status in ("pending", "queued", "running") for d in deps):
                continue
            if all(d is not None and d.status == "done" for d in deps) or t.tolerate_failures:
                out.append(t)
            else:
                t.status, t.error = "skipped", "a dependency failed"
        return out


class Supervisor:
    def __init__(
        self,
        graph: TaskGraph,
        *,
        concurrency: int = 4,
        store: Store | None = None,
        run_id: str | None = None,
        before_task: Callable[[Task], None] | None = None,
    ) -> None:
        self.graph = graph
        self.concurrency = concurrency
        self.store = store
        self.run_id = run_id
        self.before_task = before_task
        self.aborted: str | None = None
        self._resumed = store.checkpoints(run_id) if store and run_id else {}
        self._counter = itertools.count()

    def _checkpoint(self, t: Task) -> None:
        if self.store and self.run_id:
            payload = t.dump(t.result) if t.status == "done" else {"error": t.error}
            self.store.checkpoint(self.run_id, t.id, t.kind, t.status, payload)

    def _finish(self, t: Task) -> None:
        if t.status == "done" and t.spawn is not None:
            for child in t.spawn(t.result):
                self.graph.add(child)

    def _run_task(self, t: Task) -> Any:
        with tracing.span(
            f"task {t.kind}",
            "stage" if t.kind in ("ingest", "execute", "triage", "report") else "agent",
            **{
                "agentqa.task_id": t.id,
                "agentqa.task_kind": t.kind,
                "agentqa.priority": t.priority,
            },
        ):
            return t.fn()

    def run(self) -> None:
        queue: list[tuple[int, int, Task]] = []
        running: dict[Future[Any], Task] = {}
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            while True:
                for t in self.graph.ready():
                    if t.resumable and t.id in self._resumed and self._resumed[t.id][1] == "done":
                        t.result, t.status = t.load(self._resumed[t.id][2]), "done"
                        tracing.event("checkpoint.resumed", task_id=t.id)
                        self._finish(t)
                        continue
                    if self.aborted and not t.always_run:
                        t.status, t.error = "skipped", f"run aborted: {self.aborted}"
                        continue
                    t.status = "queued"
                    heapq.heappush(queue, (-t.priority, next(self._counter), t))
                while queue and len(running) < self.concurrency:
                    _, _, t = heapq.heappop(queue)
                    if self.aborted and not t.always_run:
                        t.status, t.error = "skipped", f"run aborted: {self.aborted}"
                        continue
                    try:
                        if self.before_task and not t.always_run:
                            self.before_task(t)
                    except (KillSwitchEngaged, BudgetExceeded) as exc:
                        self.aborted = f"{type(exc).__name__}: {exc}"
                        tracing.event("run.aborted", reason=self.aborted, at_task=t.id)
                        t.status, t.error = "skipped", f"run aborted: {self.aborted}"
                        continue
                    t.status = "running"
                    ctx = contextvars.copy_context()  # carry the trace and run_id into the worker
                    running[pool.submit(ctx.run, self._run_task, t)] = t
                if not running:
                    if not queue and not self.graph.ready():
                        break
                    continue
                done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for fut in done:
                    t = running.pop(fut)
                    try:
                        t.result, t.status = fut.result(), "done"
                    except (KillSwitchEngaged, BudgetExceeded) as exc:
                        self.aborted = f"{type(exc).__name__}: {exc}"
                        t.status, t.error = "failed", self.aborted
                        tracing.event("run.aborted", reason=self.aborted, at_task=t.id)
                    except Exception as exc:  # failure isolation: record and continue
                        t.status, t.error = "failed", f"{type(exc).__name__}: {exc}"
                        log.warning("task_failed", task_id=t.id, error=t.error)
                        tracing.event("task.failed", task_id=t.id, error=t.error[:300])
                    self._checkpoint(t)
                    self._finish(t)
