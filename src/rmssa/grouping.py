"""Stage 3 of (M)SSA — grouping support and separability diagnostics.

Grouping itself is just choosing index sets (done by the caller / orchestrator); the
non-trivial parts are the *diagnostics* that guide it:

  * the w-correlation matrix between reconstructed components (near-zero off-diagonal
    => well separated; large => the components belong together), and
  * helpers to build the elementary component series and to auto-suggest groups.

See report/ssa_math_notes.md for the weight definition.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .decomposition import Decomposition
from .reconstruction import diagonal_average

__all__ = [
    "wcorr_weights",
    "wcorrelation_matrix",
    "elementary_series",
    "elementary_wcorrelation",
    "suggest_groups_by_contribution",
]


def wcorr_weights(n: int, window: int) -> np.ndarray:
    """Anti-diagonal weights w_s = #{(i, j): i + j = s} = min(s+1, L*, N - s).

    L* = min(L, K). These are the lengths of the anti-diagonals of the trajectory
    matrix and define the inner product under which SSA separability is measured.
    """
    k = n - window + 1
    lstar = min(window, k)  # longest anti-diagonal length
    s = np.arange(n)
    return np.minimum.reduce([s + 1, np.full(n, lstar), n - s])


def wcorrelation_matrix(series_stack: Sequence[np.ndarray] | np.ndarray, window: int) -> np.ndarray:
    """Weighted-correlation matrix between a set of reconstructed series.

    Parameters
    ----------
    series_stack : (m, N) array (or sequence of m length-N series).
    window : window length L used in the embedding (sets the weights).

    Returns
    -------
    (m, m) symmetric matrix with unit diagonal; entry (a, b) is the w-correlation
    between series a and b.
    """
    F = np.atleast_2d(np.asarray(series_stack, dtype=float))
    m, n = F.shape
    w = wcorr_weights(n, window)
    # weighted inner products
    G = (F * w) @ F.T  # (m, m) Gram matrix of weighted inner products
    norms = np.sqrt(np.diag(G))
    denom = np.outer(norms, norms)
    with np.errstate(invalid="ignore", divide="ignore"):
        W = np.where(denom > 0, G / denom, 0.0)
    np.fill_diagonal(W, 1.0)
    return W


def elementary_series(decomposition: Decomposition, n_components: int | None = None) -> np.ndarray:
    """Reconstruct the leading elementary components as individual series.

    Returns a (m, N) array where row i is the diagonal-average of the i-th elementary
    matrix X_i = s_i U_i V_i^T. For MSSA this treats the stacked matrix as one block;
    use per-channel reconstruction for channel-level diagnostics.
    """
    m = decomposition.rank if n_components is None else min(n_components, decomposition.rank)
    return np.vstack([diagonal_average(decomposition.elementary(i)) for i in range(m)])


def elementary_wcorrelation(
    decomposition: Decomposition, window: int, n_components: int | None = None
) -> np.ndarray:
    """w-correlation matrix among the leading elementary components."""
    series = elementary_series(decomposition, n_components)
    return wcorrelation_matrix(series, window)


def suggest_groups_by_contribution(
    decomposition: Decomposition, threshold: float = 0.99
) -> dict[str, list[int]]:
    """A naive automatic grouping: leading components up to ``threshold`` cumulative
    variance form the "signal" group, the remainder form "noise".

    A convenience starting point only -- real grouping uses the w-correlation matrix
    and the scree/phase diagnostics. Returns {"signal": [...], "noise": [...]}.
    """
    contrib = decomposition.contributions()
    csum = np.cumsum(contrib)
    cutoff = int(np.searchsorted(csum, threshold) + 1)
    cutoff = min(cutoff, decomposition.rank)
    return {
        "signal": list(range(cutoff)),
        "noise": list(range(cutoff, decomposition.rank)),
    }
