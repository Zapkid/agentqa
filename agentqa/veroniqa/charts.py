"""The home page's usage and cost charts, as plain HTML and CSS (the site's CSP allows no scripts).

The numbers are the benchmark means from results/*.json (3 cold runs per strategy, simulated
models priced at each model's list price). They are constants because the deployed image does not
ship results/; tests/test_veroniqa_charts.py recomputes every one from the result files, so they
cannot drift. Each chart has a hover tooltip per bar and a table view with the same numbers.
"""

from __future__ import annotations

import html

# Strategy: (code, name, list-equivalent $ per run, bug recall %, tokens per stage per run)
STRATEGIES = [
    ("S0", "Strong model for everything", 0.473, 92, (25653, 42085, 18498, 4864)),
    ("S1", "Cheap model for everything", 0.037, 47, (25287, 40861, 7753, 624)),
    ("S2", "Static role routing", 0.173, 67, (25653, 61723, 11843, 1386)),
    ("S3", "Smart dispatcher (AgentQA)", 0.221, 94, (18297, 43572, 20809, 3752)),
]
HIGHLIGHT = "S3"
STAGES = ["Planner", "Generator", "Triage", "Judge"]

# Memory: the same suite run three times on one store (seed 0)
MEMORY = [
    ("Run 1", "cold start", 105542, 0.248),
    ("Run 2", "with 9 lessons from run 1", 98737, 0.214),
    ("Run 3", "lessons plus diff-aware reuse", 38899, 0.071),
]

CSS = """
.charts { display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr)); }
.chart { margin:0; background:var(--surface); border:1px solid var(--line); border-radius:12px;
         padding:22px 22px 18px; display:flex; flex-direction:column; gap:16px; }
.chart.wide { grid-column:1/-1; }
.chart figcaption { margin:0; color:var(--fg); font-size:inherit; }
.chart h3 { margin:0 0 4px; font-size:17px; }
.chart .sub { margin:0; color:var(--muted); font-size:14px; }
.rows { list-style:none; margin:0; padding:0; display:grid; gap:14px; }
.row-label { display:flex; justify-content:space-between; gap:12px; font-size:13.5px;
             color:var(--soft); margin-bottom:6px; }
.row-label .note { color:var(--muted); white-space:nowrap; }
.track { display:flex; align-items:center; gap:10px; border-left:1px solid #383835; min-height:16px; }
.bar { display:flex; gap:2px; height:16px; min-width:3px; }
.bar span { height:100%; background:var(--bar,#5a5a70); }
.bar span:last-child { border-radius:0 4px 4px 0; }
.bar.hl span { --bar:var(--accent); }
.val { font-size:13px; color:var(--fg); white-space:nowrap; font-variant-numeric:tabular-nums; }
.legend { display:flex; flex-wrap:wrap; gap:6px 16px; margin:0; padding:0; list-style:none;
          font-size:13px; color:var(--soft); }
.legend li { display:flex; align-items:center; gap:7px; }
.legend i { width:10px; height:10px; border-radius:2px; background:var(--bar); }
.s1 { --bar:#3987e5; } .s2 { --bar:#d95926; } .s3 { --bar:#199e70; } .s4 { --bar:#c98500; }
.chart details { background:none; border:0; padding:0; margin:0; }
.chart summary { font-size:13px; font-weight:500; color:var(--accent-2); justify-content:flex-start; }
.chart summary::after { content:none; }
.chart table { border-collapse:collapse; width:100%; font-size:13px; margin-top:10px; }
.chart th, .chart td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line);
                       color:var(--soft); font-variant-numeric:tabular-nums; }
.chart th { color:var(--muted); font-weight:600; }
.chart .table-wrap { overflow-x:auto; }
.charts-note { color:var(--muted); font-size:13px; margin:14px 0 0; }
"""

MAX_WIDTH = 76  # % of the track a full bar takes, leaving room for its value label


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _k(n: float) -> str:
    return f"{n / 1000:.0f}K" if n >= 10_000 else f"{n / 1000:.1f}K"


def _pct(value: float, top: float) -> str:
    return f"{value / top * MAX_WIDTH:.2f}%"


def _table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th scope='col'>{_e(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in r) + "</tr>" for r in rows)
    return (
        "<details><summary>Show as table</summary><div class='table-wrap'><table>"
        f"<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div></details>"
    )


def cost_chart() -> str:
    top = max(cost for _, _, cost, _, _ in STRATEGIES)
    rows = []
    for code, name, cost, recall, _ in STRATEGIES:
        hl = " hl" if code == HIGHLIGHT else ""
        tip = f"{code} {name}: ${cost:.3f} per run, {recall}% of planted bugs found"
        rows.append(
            f'<li title="{_e(tip)}"><div class="row-label"><span>{_e(code)} · {_e(name)}</span>'
            f'<span class="note">{recall}% of bugs found</span></div>'
            f'<div class="track"><div class="bar{hl}" style="width:{_pct(cost, top)}"><span style="flex:1">'
            f'</span></div><span class="val">${cost:.3f}</span></div></li>'
        )
    s0 = next(c for code, _, c, _, _ in STRATEGIES if code == "S0")
    s3 = next(c for code, _, c, _, _ in STRATEGIES if code == HIGHLIGHT)
    table = _table(
        ["Strategy", "$ per run (list price)", "Bugs found"],
        [[f"{c} {n}", f"${cost:.3f}", f"{r}%"] for c, n, cost, r, _ in STRATEGIES],
    )
    return f"""<figure class="chart" aria-labelledby="chart-cost">
  <figcaption><h3 id="chart-cost">Cost per test run</h3>
  <p class="sub">The smart dispatcher finds the most bugs at {s3 / s0:.0%} of the cost of using the
  strongest model for everything.</p></figcaption>
  <ul class="rows">{"".join(rows)}</ul>
  {table}
</figure>"""


def usage_chart() -> str:
    totals = [sum(stages) for *_, stages in STRATEGIES]
    top = max(totals)
    legend = "".join(f'<li class="s{i}"><i></i>{_e(s)}</li>' for i, s in enumerate(STAGES, 1))
    rows = []
    for (code, name, _, _, stages), total in zip(STRATEGIES, totals, strict=True):
        segs = "".join(
            f'<span class="s{i}" style="flex:{n}" title="{_e(stage)}: {n:,} tokens"></span>'
            for i, (stage, n) in enumerate(zip(STAGES, stages, strict=True), 1)
        )
        rows.append(
            f'<li title="{_e(code)} {_e(name)}: {total:,} tokens per run">'
            f'<div class="row-label"><span>{_e(code)} · {_e(name)}</span></div>'
            f'<div class="track"><div class="bar" style="width:{_pct(total, top)}">{segs}</div>'
            f'<span class="val">{_k(total)}</span></div></li>'
        )
    table = _table(
        ["Strategy", *STAGES, "Total"],
        [[f"{c} {n}", *(f"{x:,}" for x in st), f"{sum(st):,}"] for c, n, _, _, st in STRATEGIES],
    )
    s0 = next(t for (c, *_), t in zip(STRATEGIES, totals, strict=True) if c == "S0")
    s3 = next(t for (c, *_), t in zip(STRATEGIES, totals, strict=True) if c == HIGHLIGHT)
    return f"""<figure class="chart" aria-labelledby="chart-usage">
  <figcaption><h3 id="chart-usage">Tokens per test run, by stage</h3>
  <p class="sub">The dispatcher uses about as many tokens as the strong model ({_k(s3)} vs
  {_k(s0)}). It saves money by giving most of the work to cheaper models, not by doing less.</p>
  </figcaption>
  <ul class="legend" aria-label="Stages">{legend}</ul>
  <ul class="rows">{"".join(rows)}</ul>
  {table}
</figure>"""


def memory_chart() -> str:
    top = max(tokens for _, _, tokens, _ in MEMORY)
    rows = []
    for i, (run, what, tokens, cost) in enumerate(MEMORY):
        hl = " hl" if i == len(MEMORY) - 1 else ""
        rows.append(
            f'<li title="{_e(run)}, {_e(what)}: {tokens:,} tokens, ${cost:.3f}">'
            f'<div class="row-label"><span>{_e(run)} · {_e(what)}</span>'
            f'<span class="note">${cost:.3f}</span></div>'
            f'<div class="track"><div class="bar{hl}" style="width:{_pct(tokens, top)}">'
            f'<span style="flex:1"></span></div><span class="val">{_k(tokens)} tokens</span></div></li>'
        )
    first, last = MEMORY[0], MEMORY[-1]
    table = _table(
        ["Run", "Tokens", "$ (list price)"],
        [[f"{r}, {w}", f"{t:,}", f"${c:.3f}"] for r, w, t, c in MEMORY],
    )
    return f"""<figure class="chart wide" aria-labelledby="chart-memory">
  <figcaption><h3 id="chart-memory">Repeat runs get cheaper</h3>
  <p class="sub">The same suite run three times with memory: the third run cost
  {1 - last[3] / first[3]:.0%} less than the first, at the same recall.</p></figcaption>
  <ul class="rows">{"".join(rows)}</ul>
  {table}
</figure>"""


def charts_html() -> str:
    return f'<div class="charts">{cost_chart()}{usage_chart()}{memory_chart()}</div>'


def charts_markdown() -> str:
    cost = "\n".join(f"| {c} {n} | ${x:.3f} | {r}% |" for c, n, x, r, _ in STRATEGIES)
    usage = "\n".join(
        f"| {c} {n} | " + " | ".join(f"{v:,}" for v in st) + f" | {sum(st):,} |"
        for c, n, _, _, st in STRATEGIES
    )
    memory = "\n".join(f"| {r}, {w} | {t:,} | ${c:.3f} |" for r, w, t, c in MEMORY)
    return f"""### Cost per test run

| strategy | $ per run (list price) | bugs found |
|---|---|---|
{cost}

### Tokens per test run, by stage

| strategy | {" | ".join(s.lower() for s in STAGES)} | total |
|---|---|---|---|---|---|
{usage}

### Repeat runs get cheaper (memory)

| run | tokens | $ (list price) |
|---|---|---|
{memory}
"""
