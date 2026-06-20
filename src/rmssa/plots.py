"""Diagnostic plots for (M)SSA: scree, w-correlation heatmap, component series.

Uses a non-interactive backend so figures render in headless / CI runs. Each function
returns the Matplotlib Figure and optionally saves it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["plot_scree", "plot_wcorrelation", "plot_components", "savefig"]


def savefig(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scree(contributions: np.ndarray, n: int = 30, title: str = "Scree plot"):
    contributions = np.asarray(contributions)[:n]
    fig, ax = plt.subplots(figsize=(7, 4))
    idx = np.arange(1, contributions.size + 1)
    ax.semilogy(idx, 100 * contributions, "o-", ms=4)
    ax.set_xlabel("component index")
    ax.set_ylabel("variance contribution (%)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    return fig


def plot_wcorrelation(W: np.ndarray, title: str = "w-correlation matrix"):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(np.abs(W), cmap="viridis", vmin=0, vmax=1, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("component")
    ax.set_ylabel("component")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|w-corr|")
    return fig


def plot_components(
    original: np.ndarray,
    components: Mapping[str, np.ndarray],
    title: str = "Reconstructed components",
    max_points: int = 1000,
):
    """Overlay the original series with named reconstructed components (univariate)."""
    original = np.asarray(original).ravel()
    s = slice(0, min(len(original), max_points))
    rows = 1 + len(components)
    fig, axes = plt.subplots(rows, 1, figsize=(9, 1.8 * rows), sharex=True)
    axes = np.atleast_1d(axes)
    axes[0].plot(original[s], color="k", lw=0.8)
    axes[0].set_ylabel("original")
    axes[0].set_title(title)
    for ax, (label, comp) in zip(axes[1:], components.items()):
        comp = np.asarray(comp).ravel()
        ax.plot(comp[s], lw=0.9)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    return fig
