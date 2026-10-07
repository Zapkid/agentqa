"""Latency-vs-load and throughput-vs-load charts with the knee marked (matplotlib, PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentqa.perf.analysis import curve, knee, summarize
from agentqa.perf.models import PerfRun

# Validated categorical slots 1-2 of the dataviz reference palette (light surface #fcfcfb).
COLORS = {"baseline": "#2a78d6", "candidate": "#eb6834"}
SURFACE, TEXT, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def load_charts(runs: dict[str, PerfRun], out: Path, title: str) -> Path:
    """runs: {"baseline": run, "candidate": run} (either may be missing)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for role, run in runs.items():
        users, rps, p95 = curve(summarize(run))
        if not users:
            continue
        color = COLORS.get(role, "#555")
        k = knee(users, rps, p95)
        ax1.plot(
            users,
            p95,
            marker="o",
            markersize=6,
            linewidth=2,
            color=color,
            label=f"{role} ({run.label})",
        )
        ax2.plot(
            users,
            rps,
            marker="o",
            markersize=6,
            linewidth=2,
            color=color,
            label=f"{role} ({run.label})",
        )
        if k["latency_knee_users"]:
            i = users.index(k["latency_knee_users"])
            ax1.axvline(users[i], color=color, linestyle=":", linewidth=1)
            ax1.annotate(
                "latency knee",
                (users[i], p95[i]),
                textcoords="offset points",
                xytext=(4, 6),
                fontsize=8,
                color=MUTED,
            )
        if k["throughput_knee_users"]:
            i = users.index(k["throughput_knee_users"])
            ax2.axvline(users[i], color=color, linestyle=":", linewidth=1)
            label = (
                "saturated from first step" if k.get("saturated_from_start") else "throughput knee"
            )
            ax2.annotate(
                label,
                (users[i], rps[i]),
                textcoords="offset points",
                xytext=(4, -12),
                fontsize=8,
                color=MUTED,
            )
    ax1.set(title="p95 latency vs load", xlabel="concurrent users", ylabel="p95 (ms)")
    ax2.set(title="Throughput vs load", xlabel="concurrent users", ylabel="requests / s")
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(alpha=0.25, color=MUTED, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.title.set_color(TEXT)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=11, color=TEXT)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
