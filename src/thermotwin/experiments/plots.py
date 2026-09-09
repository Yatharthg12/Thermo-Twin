"""Offline colorblind-conscious PNG/SVG experiment figures."""

from __future__ import annotations

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442"]


def line_plot(frame: pd.DataFrame, x: str, y: str, group: str, title: str, xlabel: str, ylabel: str, output_base: str | Path) -> list[Path]:
    """Save an accessible labeled line figure in PNG and SVG forms."""
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    for index, (name, subset) in enumerate(frame.groupby(group, sort=False)):
        ordered = subset.sort_values(x)
        axis.plot(ordered[x], ordered[y], marker="o", linewidth=2, label=str(name), color=COLORS[index % len(COLORS)])
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axis.grid(alpha=.22)
    axis.legend(frameon=False, fontsize=8)
    paths = []
    for suffix in (".png", ".svg"):
        path = base.with_suffix(suffix)
        fig.savefig(path, dpi=150)
        paths.append(path)
    plt.close(fig)
    return paths


def bar_plot(frame: pd.DataFrame, category: str, value: str, title: str, ylabel: str, output_base: str | Path) -> list[Path]:
    """Save a labeled categorical comparison in PNG and SVG forms."""
    base = Path(output_base)
    fig, axis = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    axis.bar(frame[category].astype(str), frame[value], color=COLORS[: len(frame)])
    axis.set(title=title, ylabel=ylabel)
    axis.tick_params(axis="x", rotation=24)
    axis.grid(axis="y", alpha=.22)
    paths = []
    for suffix in (".png", ".svg"):
        path = base.with_suffix(suffix)
        fig.savefig(path, dpi=150)
        paths.append(path)
    plt.close(fig)
    return paths

