"""Seeded-defect toggles read from the environment at import/startup time."""

from __future__ import annotations

import os


def _ids(var: str) -> frozenset[str]:
    return frozenset(x.strip().upper() for x in os.environ.get(var, "").split(",") if x.strip())


def on(bug_id: str) -> bool:
    return bug_id in _ids("BUGS")


def perf(bug_id: str) -> bool:
    return bug_id in _ids("PERF_BUGS")


def active() -> dict[str, list[str]]:
    return {"bugs": sorted(_ids("BUGS")), "perf_bugs": sorted(_ids("PERF_BUGS"))}
