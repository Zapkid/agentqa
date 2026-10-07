"""Ground truth by differential testing (no manual labelling).

A generated suite is executed (executor only, no LLM) against the clean build, every single-bug
build and the all-bugs build:

- bug Bi is detected  <=> some test passes on clean and fails on the Bi build
- a test is a test_bug     <=> it fails on the clean build
- a test is a product_bug  <=> it passes on clean, fails on the all-bugs build and fails on at least
                               one single-bug build
- a test is flaky          <=> its outcome changes across reruns of the same build
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from agentqa.config import REPO_ROOT
from agentqa.executor.runner import Executor
from agentqa.models import RunResult, ValidatedTest
from agentqa.target_config import TargetConfig
from target_api.launcher import TargetServer

BUGS = yaml.safe_load((REPO_ROOT / "target_api" / "bugs.yaml").read_text(encoding="utf-8"))
BUG_IDS = [b["id"] for b in BUGS]
BUG_CATEGORY = {b["id"]: b["category"] for b in BUGS}

FAIL = ("failed", "error")


@dataclass
class GroundTruth:
    outcomes: dict[str, dict[str, str]] = field(default_factory=dict)  # build -> test -> outcome
    flaky_tests: set[str] = field(default_factory=set)
    clean_run: RunResult | None = None

    def passes_clean(self, test: str) -> bool:
        return self.outcomes["clean"].get(test) == "passed"

    def detected(self) -> dict[str, list[str]]:
        """bug id -> tests that detect it."""
        out: dict[str, list[str]] = {}
        for bug in BUG_IDS:
            hits = [
                t
                for t, o in self.outcomes.get(bug, {}).items()
                if o in FAIL and self.passes_clean(t)
            ]
            if hits:
                out[bug] = hits
        return out

    def label(self, test: str) -> str:
        if test in self.flaky_tests:
            return "flaky"
        if self.outcomes["clean"].get(test) in FAIL:
            return "test_bug"
        fails_all = self.outcomes.get("all", {}).get(test) in FAIL
        fails_single = any(self.outcomes.get(b, {}).get(test) in FAIL for b in BUG_IDS)
        if fails_all and fails_single:
            return "product_bug"
        if fails_all:
            return (
                "interaction"  # fails only when bugs combine (e.g. one bug masks another's setup)
            )
        return "pass"

    def validity_rate(self) -> float:
        clean = self.outcomes["clean"]
        considered = [o for o in clean.values() if o != "skipped"]
        return sum(o == "passed" for o in considered) / len(considered) if considered else 0.0


def build_outcomes(run: RunResult) -> tuple[dict[str, str], set[str]]:
    outcomes: dict[str, str] = {r.test_name: str(r.outcome) for r in run.results}
    flaky = {r.test_name for r in run.results if r.flaky}
    return outcomes, flaky


def differential(
    tests: list[ValidatedTest],
    target: TargetConfig,
    work_dir: Path,
    all_bugs_run: RunResult | None = None,
    parallel: int = 4,
) -> GroundTruth:
    """Execute the suite on clean (with flaky reruns) and on every single-bug build (no reruns)."""
    gt = GroundTruth()

    def run_build(label: str, bugs: list[str], reruns: int) -> tuple[str, RunResult]:
        with TargetServer(bugs=bugs) as srv:
            return label, Executor(target, reruns=reruns).run(tests, srv.base_url, work_dir / label)

    jobs: list[tuple[str, list[str], int]] = [("clean", [], 3)] + [(b, [b], 0) for b in BUG_IDS]
    if all_bugs_run is None:
        jobs.append(("all", list(BUG_IDS), 0))
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for label, run in pool.map(lambda j: run_build(*j), jobs):
            outcomes, flaky = build_outcomes(run)
            gt.outcomes[label] = outcomes
            if label == "clean":
                gt.flaky_tests |= flaky
                gt.clean_run = run
    if all_bugs_run is not None:
        gt.outcomes["all"], flaky_all = build_outcomes(all_bugs_run)
        gt.flaky_tests |= flaky_all
    return gt
