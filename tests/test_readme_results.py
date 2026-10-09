"""The README's Results block is generated from results/*.json and must not drift from it."""

from __future__ import annotations

from agentqa.evals.report import BEGIN, END, README, latest_results, readme_results_from_files


def test_readme_results_block_matches_result_files() -> None:
    text = README.read_text(encoding="utf-8")
    block = text[text.index(BEGIN) + len(BEGIN) : text.index(END)].strip()
    expected = readme_results_from_files(*latest_results()).strip()
    assert block == expected, "README results are stale: run `agentqa eval` (or update_readme)"


def test_opener_headline_numbers_match_result_files() -> None:
    import json

    func_path, perf_path = latest_results()
    assert func_path and perf_path
    st = json.loads(func_path.read_text())["strategies"]
    cases = json.loads(perf_path.read_text())["cases"]
    s0, s3 = st["S0"]["aggregate"], st["S3"]["aggregate"]
    ratio = s3["cost_usd_list_equivalent"]["mean"] / s0["cost_usd_list_equivalent"]["mean"]
    flagged = sum(bool(c["regressed"]) for c in cases)
    opener = README.read_text(encoding="utf-8").split("## In technical terms")[0]
    text = " ".join(opener.split())
    for phrase in (
        f"found {s3['recall']['mean'] * 100:.0f}% of the planted bugs",
        f"against {s0['recall']['mean'] * 100:.0f}% for always using the strongest model",
        f"at {ratio:.0%} of the cost",
        f"caught {flagged} of {len(cases)} performance defects",
    ):
        assert phrase in text, f"opener out of date: expected {phrase!r}"
