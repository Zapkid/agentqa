"""The promo-video storyboards build from the committed data and draw frames (no encoding)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts/videos"))

import engine  # noqa: E402
import render  # noqa: E402


@pytest.mark.parametrize("key", sorted(render.VIDEOS))
def test_storyboards_draw_every_scene(key: str) -> None:
    func, perf = render.load()
    _, build = render.VIDEOS[key]
    scenes = build(func, perf)
    assert 4 <= len(scenes) <= 7 and 20 <= sum(d for d, _ in scenes) <= 40
    for dur, draw in scenes:
        for t in (0.0, dur / 2, dur):
            c = engine.Canvas.new()
            draw(c, t, min(1.0, t / dur))
            assert c.img.size == (engine.W, engine.H)


def test_on_screen_numbers_come_from_the_results() -> None:
    func, _ = render.load()
    st = func["strategies"]
    ratio = (
        st["S3"]["aggregate"]["cost_usd_list_equivalent"]["mean"]
        / st["S0"]["aggregate"]["cost_usd_list_equivalent"]["mean"]
    )
    source = (ROOT / "scripts/videos/render.py").read_text()
    assert "47%" not in source and "0.473" not in source  # nothing hard-coded
    assert f"{ratio:.0%}" == "47%"  # what the cost video shows, computed


def test_veroniqa_facts_match_screenshots() -> None:
    facts = json.loads((ROOT / "media/veroniqa/run.json").read_text())
    assert facts["simulated"] is True and facts["outcomes"]["failed"] == len(facts["product_bugs"])
    for name in ("knowledge", "chat", "ssrf", "runs"):
        assert (ROOT / f"media/veroniqa/{name}.png").exists()
