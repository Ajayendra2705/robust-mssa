"""Stage 1 of (M)SSA — embedding.

Builds the (block-)Hankel trajectory matrix that the decomposition step operates on.

Definitions (0-based indices), see report/ssa_math_notes.md:

    Univariate:  series F = (f_0, ..., f_{N-1}), window length L (1 < L < N),
                 K = N - L + 1. The trajectory matrix X in R^{L x K} has
                     X[i, j] = f_{i + j}
                 so it is Hankel (constant along anti-diagonals).

    Multivariate (horizontal / common-window MSSA): for p series with a shared
    window L, concatenate the per-channel trajectory matrices column-wise:
                     H = [X^(1) | X^(2) | ... | X^(p)]  in R^{L x K_tot}.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = ["trajectory_matrix", "mssa_trajectory_matrix", "embed", "n_columns"]


def n_columns(n: int, window: int) -> int:
    """Number of trajectory columns K = N - L + 1."""
    return n - window + 1


def _validate_window(n: int, window: int) -> None:
    if not isinstance(window, (int, np.integer)):
        raise TypeError(f"window length L must be an int, got {type(window)!r}")
    if not (1 < window < n):
        raise ValueError(
            f"window length L must satisfy 1 < L < N (got L={window}, N={n})"
        )


def trajectory_matrix(series: Sequence[float] | np.ndarray, window: int) -> np.ndarray:
    """Univariate trajectory (Hankel) matrix.

    Parameters
    ----------
    series : 1-D array-like of length N.
    window : window length L, with 1 < L < N.

    Returns
    -------
    X : ndarray of shape (L, K), K = N - L + 1, with X[i, j] = series[i + j].
        A fresh, contiguous, writable array (not a view onto ``series``).
    """
    x = np.asarray(series, dtype=float).ravel()
    n = x.shape[0]
    _validate_window(n, window)
    k = n_columns(n, window)
    # Hankel via anti-diagonal index broadcasting: idx[i, j] = i + j.
    idx = np.arange(window)[:, None] + np.arange(k)[None, :]
    return np.ascontiguousarray(x[idx])


def mssa_trajectory_matrix(
    series: Iterable[Sequence[float] | np.ndarray],
    window: int,
) -> tuple[np.ndarray, list[int]]:
    """Horizontal (common-window) MSSA block-Hankel trajectory matrix.

    Parameters
    ----------
    series : iterable of p 1-D series. Series may have different lengths N_j;
             each contributes a block of width K_j = N_j - L + 1.
    window : shared window length L.

    Returns
    -------
    H : ndarray of shape (L, K_tot), K_tot = sum_j K_j, the column-wise
        concatenation of the per-channel trajectory matrices.
    widths : list[int] of the per-channel column counts K_j (block boundaries),
        needed to split reconstructions back into channels in later phases.
    """
    blocks: list[np.ndarray] = []
    widths: list[int] = []
    for j, s in enumerate(series):
        block = trajectory_matrix(s, window)  # validates per channel
        if block.shape[0] != window:  # defensive; trajectory_matrix guarantees this
            raise ValueError(f"channel {j}: expected L={window} rows, got {block.shape[0]}")
        blocks.append(block)
        widths.append(block.shape[1])
    if not blocks:
        raise ValueError("series is empty: need at least one channel")
    return np.concatenate(blocks, axis=1), widths


# Convenience alias used in notes / READMEs.
def embed(series, window: int):
    """Alias for :func:`trajectory_matrix` (univariate embedding)."""
    return trajectory_matrix(series, window)
