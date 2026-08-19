"""Publication-oriented matplotlib style for EAF-MAS v2 figures."""

from __future__ import annotations

from typing import Iterable


COLORS = {
    "raw": "#374151",
    "adjusted": "#f97316",
    "observed": "#111827",
    "beneficial": "#2f9e44",
    "harmful": "#c92a2a",
    "neutral": "#6b7280",
    "abstain": "#9ca3af",
    "pass": "#55a868",
    "weak": "#d4a72c",
    "fail": "#c44e52",
    "missing": "#d1d5db",
    "audit": "#4c78a8",
    "memory": "#7c3aed",
}


def apply_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": "#d1d5db",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_labels(axes: Iterable, labels: str = "abcdefghijklmnopqrstuvwxyz") -> None:
    for ax, label in zip(axes, labels):
        ax.text(
            -0.08,
            1.06,
            f"({label})",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
        )
