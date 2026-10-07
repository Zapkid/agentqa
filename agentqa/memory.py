"""F8 Memory.

Within a run: the cascade's feedback history ("what I already tried") travels with each task.
Across runs: a SQLite lessons store. Lessons are short, factual notes derived from verified
outcomes (grounding rejections, test bugs, flaky tests), retrieved per endpoint into the
planner and generator prompts. Lessons come only from our own run history, never from
customer documents, so they are trusted context.
"""

from __future__ import annotations

import time

from agentqa.models import Finding
from agentqa.obs import tracing
from agentqa.store import Store


class LessonStore:
    def __init__(self, store: Store, api_key: str) -> None:
        self.store = store
        self.api_key = api_key

    def add(self, endpoint: str, kind: str, text: str, run_id: str) -> None:
        self.store.execute(
            "INSERT OR IGNORE INTO lessons (api_key, endpoint, kind, text, source_run, created)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (self.api_key, endpoint, kind, text[:400], run_id, time.time()),
        )

    def for_endpoint(self, endpoint: str, limit: int = 5) -> list[str]:
        rows = self.store.query(
            "SELECT id, text FROM lessons WHERE api_key=? AND endpoint IN (?, '*')"
            " ORDER BY created DESC LIMIT ?",
            (self.api_key, endpoint, limit),
        )
        for r in rows:
            self.store.execute("UPDATE lessons SET uses = uses + 1 WHERE id = ?", (r["id"],))
        if rows:
            tracing.event("memory.lessons_used", endpoint=endpoint, count=len(rows))
        return [r["text"] for r in rows]

    def count(self) -> int:
        return int(
            self.store.query("SELECT COUNT(*) AS c FROM lessons WHERE api_key=?", (self.api_key,))[
                0
            ]["c"]
        )


def derive_lessons(
    lessons: LessonStore,
    run_id: str,
    rejections: list[tuple[str, str, str]],
    findings: list[Finding],
) -> int:
    """rejections: (endpoint, violation_kind, detail) seen before repair in this run."""
    before = lessons.count()
    for endpoint, kind, detail in rejections:
        if kind in ("unknown_path", "unknown_field", "undocumented_status", "method_not_allowed"):
            lessons.add(
                endpoint,
                "grounding",
                f"{endpoint}: a generated test was rejected ({kind}: {detail[:160]}). "
                "Use only the paths, fields and status codes in the spec.",
                run_id,
            )
        elif kind == "fails_on_reference":
            lessons.add(
                endpoint,
                "validity",
                f"{endpoint}: a generated test failed on the reference build "
                f"({detail[:160]}). Re-read the requirement before asserting values or statuses.",
                run_id,
            )
    for f in findings:
        if f.classification == "test_bug":
            lessons.add(
                f.endpoint,
                "test_bug",
                f"{f.endpoint}: a test was wrong: {f.root_cause_hypothesis[:200]}",
                run_id,
            )
        elif f.classification == "flaky":
            lessons.add(
                f.endpoint,
                "flaky",
                f"{f.endpoint}: test {f.test_names[0]} was flaky; avoid order- or "
                "time-dependent assertions.",
                run_id,
            )
    return lessons.count() - before
