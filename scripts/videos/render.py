"""Render AgentQA's three promo videos from measured data.

    uv run python scripts/videos/render.py            # all three, into media/videos/
    uv run python scripts/videos/render.py 2          # just one

Every number on screen comes from results/*.json (via agentqa.evals.report.latest_results) or
from media/veroniqa/run.json (a real Veroniqa run, see capture_veroniqa.py). Benchmark numbers are
from simulated models, and each video says so on screen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import (
    BLUE,
    GREEN,
    MUTED,
    ORANGE,
    PANEL,
    TEXT,
    Canvas,
    Draw,
    H,
    W,
    font,
    footer,
    ken_burns,
    overline,
    render,
    window,
)

from agentqa.config import REPO_ROOT
from agentqa.evals.report import latest_results

OUT = REPO_ROOT / "media/videos"
SHOTS = REPO_ROOT / "media/veroniqa"
REPO = "github.com/Zapkid/agentqa"


def load() -> tuple[dict[str, Any], dict[str, Any]]:
    func_path, perf_path = latest_results()
    assert func_path and perf_path, "no results/*.json: run agentqa eval first"
    return json.loads(func_path.read_text()), json.loads(perf_path.read_text())


# ---------------------------------------------------------------- shared scenes


def title(over: str, line1: str, line2: str, sub: str = "") -> Draw:
    def draw(c: Canvas, t: float, p: float) -> None:
        overline(c, over, window(t, 0.1, 0.5))
        c.text((96, 250), line1, 64, TEXT, bold=True, alpha=window(t, 0.3, 0.6))
        c.text((96, 340), line2, 64, ORANGE, bold=True, alpha=window(t, 1.0, 0.6))
        if sub:
            c.wrapped((96, 450), sub, 26, 1000, MUTED, alpha=window(t, 1.6, 0.6))

    return draw


def statement(over: str, lines: list[str]) -> Draw:
    def draw(c: Canvas, t: float, p: float) -> None:
        overline(c, over)
        y = 190.0
        for i, line in enumerate(lines):
            y = c.wrapped((96, y), line, 36 if i == 0 else 28, 1060,
                          TEXT if i == 0 else MUTED, bold=i == 0,
                          alpha=window(t, 0.2 + 0.9 * i, 0.6)) + 24  # fmt: skip

    return draw


def end_card(tagline: str, caveat: str, cta: str) -> Draw:
    def draw(c: Canvas, t: float, p: float) -> None:
        c.text((W / 2, 250), "AgentQA", 84, TEXT, bold=True, anchor="mm", alpha=window(t, 0, 0.6))
        c.text((W / 2, 335), tagline, 30, ORANGE, anchor="mm", alpha=window(t, 0.4, 0.6))
        c.text((W / 2, 420), cta, 26, TEXT, anchor="mm", alpha=window(t, 0.9, 0.6))
        c.text((W / 2, 470), REPO, 22, MUTED, anchor="mm", alpha=window(t, 0.9, 0.6))
        c.text((W / 2, H - 70), caveat, 17, MUTED, anchor="mm", alpha=window(t, 1.2, 0.6))

    return draw


def bars(over: str, heading: str, rows: list[tuple[str, float, str, tuple[int, int, int]]],
         scale: float, callout: str = "", note: str = "") -> Draw:  # fmt: skip
    """Horizontal bars that grow in, one after another."""

    def draw(c: Canvas, t: float, p: float) -> None:
        overline(c, over)
        c.text((96, 140), heading, 40, TEXT, bold=True, alpha=window(t, 0, 0.5))
        for i, (label, value, shown, color) in enumerate(rows):
            y = 250 + i * 130
            k = window(t, 0.5 + 0.8 * i, 1.4)
            c.text((96, y), label, 24, MUTED, alpha=window(t, 0.3 + 0.8 * i, 0.4))
            c.rect((96, y + 40, 96 + 860, y + 92), PANEL, radius=10)
            c.rect((96, y + 40, 96 + 860 * value / scale * k, y + 92), color, radius=10)
            c.text((980, y + 66), shown, 34, TEXT, bold=True, anchor="lm", alpha=k)
        if callout:
            c.text((96, 560), callout, 40, ORANGE, bold=True,
                   alpha=window(t, 0.6 + 0.8 * len(rows) + 0.6, 0.6))  # fmt: skip
        if note:
            footer(c, note)

    return draw


def stats(over: str, heading: str, items: list[tuple[str, str]], note: str = "") -> Draw:
    """Big numbers that appear one by one."""

    def draw(c: Canvas, t: float, p: float) -> None:
        overline(c, over)
        c.text((96, 140), heading, 40, TEXT, bold=True, alpha=window(t, 0, 0.5))
        for i, (big, small) in enumerate(items):
            y = 245 + i * 125
            a = window(t, 0.5 + 0.9 * i, 0.6)
            c.rect((96, y, 96 + 6, y + 90), GREEN, radius=3, alpha=a)
            c.text((126, y + 4), big, 50, TEXT, bold=True, alpha=a)
            c.text((126, y + 64), small, 22, MUTED, alpha=a)
        if note:
            footer(c, note)

    return draw


# ---------------------------------------------------------------- video 1: cost vs quality


def video_cost(func: dict[str, Any]) -> list[tuple[float, Draw]]:
    st = func["strategies"]
    s0, s3 = st["S0"]["aggregate"], st["S3"]["aggregate"]
    r0, r3 = s0["recall"]["mean"] * 100, s3["recall"]["mean"] * 100
    c0, c3 = s0["cost_usd_list_equivalent"]["mean"], s3["cost_usd_list_equivalent"]["mean"]
    n = s3["n"]
    note = f"Bundled benchmark, simulated models, {n} cold runs per strategy. Real models: make eval-live."
    return [
        (4.0, title("AgentQA · smart dispatch", "Same bugs found.", f"{c3 / c0:.0%} of the bill.",
                    "AI-written API tests, with the cheapest model that gets each job right.")),
        (5.5, statement("The problem", [
            "Running every test-writing job on the strongest model works. It also costs the most.",
            "AgentQA's dispatcher gives easy work to plain code and cheap models, checks every "
            "test, and escalates to the strong model only when a check fails.",
        ])),
        (6.0, bars("Quality", "Planted bugs found", [
            ("Strongest model for everything", r0, f"{r0:.0f}%", BLUE),
            ("AgentQA dispatcher", r3, f"{r3:.0f}%", ORANGE),
        ], scale=100, note=note)),
        (6.5, bars("Cost", "List-price cost per run", [
            ("Strongest model for everything", c0, f"${c0:.3f}", BLUE),
            ("AgentQA dispatcher", c3, f"${c3:.3f}", ORANGE),
        ], scale=c0 * 1.05, callout=f"{r3 - r0:+.1f} points of recall at {c3 / c0:.0%} of the cost",
            note=note)),
        (4.5, end_card("Cheapest worker that gets it right.", note,
                       "Try it: make demo")),
    ]  # fmt: skip


# ---------------------------------------------------------------- video 2: trust, measured


def video_trust(func: dict[str, Any], perf: dict[str, Any]) -> list[tuple[float, Draw]]:
    cases = perf["cases"]
    flagged = sum(bool(c["regressed"]) for c in cases)
    named = sum(bool(c["correct"]) for c in cases)
    held = perf.get("false_alarm_detail") or {}
    false_alarms = int(bool(held.get("flagged") or held.get("server_signals")))
    inj = func["injection"]
    adv = [x for x in inj["cases"] if x["label"] == "injection"]
    ben = [x for x in inj["cases"] if x["label"] != "injection"]
    m = func["memory"]
    r1, r3 = m["run1_cold"], m["run3_incremental"]
    saving = 1 - r3["cost_usd_list_equivalent"] / r1["cost_usd_list_equivalent"]
    note = "Bundled benchmark, simulated models. Injection set written by the project author."
    return [
        (4.0, title("AgentQA · evals built in", "Trust isn't a feeling.", "It's a number.",
                    "Every mechanism is scored against an API with planted defects.")),
        (6.5, stats("Performance", "Load tests that find the cause", [
            (f"{flagged}/{len(cases)} defects caught", "N+1 queries, missing index, payload bloat, "
             "blocking handler, pool exhaustion, memory leak"),
            (f"{named}/{len(cases)} root causes named", "from the target's own counters, checked "
             "against a rule-based diagnosis"),
            (f"{false_alarms} false alarms", "on a held-out clean-vs-clean repeat"),
        ], note=note)),
        (6.0, stats("Security", "Guardrails you can measure", [
            (f"{sum(x['quarantined'] for x in adv)}/{len(adv)} injections quarantined",
             "poisoned documents never reach a prompt"),
            (f"{sum(x['quarantined'] for x in ben)}/{len(ben)} benign docs flagged",
             "real requirements still get through"),
        ], note=note)),
        (6.5, bars("Memory", "Run 3 vs run 1 on the same API", [
            ("Run 1: cold", r1["tokens_total"] / 1000, f"{r1['tokens_total'] / 1000:.0f}k tokens", BLUE),
            ("Run 3: lessons + incremental reuse", r3["tokens_total"] / 1000,
             f"{r3['tokens_total'] / 1000:.0f}k tokens", ORANGE),
        ], scale=r1["tokens_total"] / 1000 * 1.05,
            callout=f"{saving:.0%} cheaper at the same recall", note=note)),
        (4.5, end_card("Observable. Guarded. Measured.", note, "Read the numbers: RESULTS.md")),
    ]  # fmt: skip


# ---------------------------------------------------------------- video 3: meet Veroniqa


def screen(
    over: str, caption: str, shot: Path, focus: tuple[float, float], badge: str = ""
) -> Draw:
    im = Image.open(shot).convert("RGB")
    box = (64, 196, W - 64, H - 34)

    def draw(c: Canvas, t: float, p: float) -> None:
        overline(c, over)
        c.wrapped((96, 116), caption, 24, 1088, TEXT, bold=True, alpha=window(t, 0, 0.5))
        frame = ken_burns(im, box, p, zoom=1.08, focus=focus)
        shadow = Image.new("RGB", (frame.width + 24, frame.height + 24), (0, 0, 0))
        c.paste(shadow.filter(ImageFilter.GaussianBlur(12)), (box[0] - 12, box[1] - 4), 0.25)
        c.paste(frame, (box[0], box[1]))
        c.draw.rounded_rectangle(box, radius=14, outline=PANEL, width=3)
        if badge and t >= 1.2:  # pops in: an alpha fade would darken it over the screenshot
            f = font(22, True)
            w = int(c.draw.textlength(badge, font=f)) + 40
            c.rect((W - 96 - w, H - 104, W - 96, H - 56), ORANGE, radius=24)
            c.text((W - 96 - w / 2, H - 80), badge, 22, (255, 255, 255), bold=True, anchor="mm")

    return draw


def video_veroniqa() -> list[tuple[float, Draw]]:
    facts = json.loads((SHOTS / "run.json").read_text())
    out = facts["outcomes"]
    quarantined = [s["name"] for s in facts["sources"] if s["quarantined"]]
    found = len(facts["product_bugs"])
    seeded = len(facts["seeded_bugs"])
    note = "Real screenshots of the app, running the simulated models on the bundled demo API."
    return [
        (4.0, title("Introducing", "Meet Veroniqa.", "Your API-testing assistant.",
                    "Projects, a knowledge base with retrieval, and AgentQA's test runs, "
                    "in one chat.")),
        (5.5, screen("1 · Knowledge", "Drop in your requirements. Every document is chunked, "
                     "scanned for prompt injection and indexed.", SHOTS / "knowledge.png",
                     (0.6, 0.3), badge=f"{len(quarantined)} poisoned doc quarantined")),
        (5.5, screen("2 · Ask", "Answers come from your own documents, with the sources "
                     "they used.", SHOTS / "chat.png", (0.5, 0.35), badge="cited, not guessed")),
        (5.0, screen("3 · Safe links", "Add web pages as knowledge. Internal and cloud-metadata "
                     "addresses are refused.", SHOTS / "ssrf.png", (0.5, 0.9),
                     badge="SSRF blocked")),
        (6.0, screen("4 · Run the tests", f"Say \"run the tests\". {out.get('passed', 0)} passed, "
                     f"{out.get('failed', 0)} failed: {found} of {seeded} seeded bugs found.",
                     SHOTS / "runs.png", (0.5, 0.45), badge=f"{facts['tokens'] / 1000:.0f}k tokens")),
        (4.5, end_card("Ask. Add knowledge. Run the tests.", note, "Try it: make veroniqa")),
    ]  # fmt: skip


VIDEOS = {
    "1": ("01-cost-vs-quality.mp4", lambda f, p: video_cost(f)),
    "2": ("02-trust-measured.mp4", lambda f, p: video_trust(f, p)),
    "3": ("03-meet-veroniqa.mp4", lambda f, p: video_veroniqa()),
}


def main(which: list[str]) -> None:
    func, perf = load()
    for key in which or list(VIDEOS):
        name, build = VIDEOS[key]
        path = render(OUT / name, build(func, perf))
        print(f"wrote {path.relative_to(REPO_ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1:])
