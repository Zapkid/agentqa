"""The home page's usage and cost charts: every number is recomputed from results/*.json."""

from __future__ import annotations

import json
import re

from agentqa.evals.report import latest_results
from agentqa.veroniqa import charts, site

STAGE_KEYS = ["planner", "generator", "triage", "judge"]


def _results() -> dict:
    func_path, _ = latest_results()
    assert func_path
    return json.loads(func_path.read_text())


def test_strategy_numbers_match_the_result_files() -> None:
    st = _results()["strategies"]
    for code, _name, cost, recall, stages in charts.STRATEGIES:
        agg, cases = st[code]["aggregate"], st[code]["cases"]
        assert cost == round(agg["cost_usd_list_equivalent"]["mean"], 3), code
        assert recall == round(agg["recall"]["mean"] * 100), code
        means = tuple(
            round(sum(c["tokens_by_stage"].get(k, 0) for c in cases) / len(cases))
            for k in STAGE_KEYS
        )
        assert stages == means, code
        assert abs(sum(stages) - agg["tokens_total"]["mean"]) <= 2, code  # rounding only
    assert [s.lower() for s in charts.STAGES] == STAGE_KEYS


def test_memory_numbers_match_the_result_files() -> None:
    mem = _results()["memory"]
    runs = [mem["run1_cold"], mem["run2_with_lessons"], mem["run3_incremental"]]
    for (_run, _what, tokens, cost), got in zip(charts.MEMORY, runs, strict=True):
        assert tokens == got["tokens_total"]
        assert cost == round(got["cost_usd_list_equivalent"], 3)
    assert f"with {mem['lessons_after_run1']} lessons" in charts.MEMORY[1][1]


def test_bars_are_proportional_and_labelled() -> None:
    page = charts.charts_html()
    widths = [float(w) for w in re.findall(r'class="bar[^"]*" style="width:([\d.]+)%"', page)]
    costs = [c for _, _, c, _, _ in charts.STRATEGIES]
    totals = [sum(s) for *_, s in charts.STRATEGIES]
    tokens = [t for _, _, t, _ in charts.MEMORY]
    expected = [
        *(v / max(vs) * charts.MAX_WIDTH for vs in (costs,) for v in vs),
        *(v / max(vs) * charts.MAX_WIDTH for vs in (totals,) for v in vs),
        *(v / max(vs) * charts.MAX_WIDTH for vs in (tokens,) for v in vs),
    ]
    assert len(widths) == len(expected)
    assert all(abs(w - e) < 0.01 for w, e in zip(widths, expected, strict=True))
    for code, _, cost, recall, _ in charts.STRATEGIES:
        assert f"${cost:.3f}" in page and f"{recall}% of bugs found" in page, code
    assert page.count("Show as table") == 3 and page.count('class="bar hl"') == 2
    for stage in charts.STAGES:
        assert f"<i></i>{stage}</li>" in page  # legend, so colour never carries identity alone


def test_charts_are_on_the_home_page_and_its_markdown() -> None:
    page, md = site.home_page(), site.home_markdown()
    assert 'id="results"' in page and 'href="/#results"' in page and charts.charts_html() in page
    assert "<script" not in charts.charts_html()  # the site's CSP allows no scripts
    assert "## Usage and cost" in md and charts.charts_markdown() in md
    assert page.count("<h1") == 1
