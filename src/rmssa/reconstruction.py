"""Stage 4 of (M)SSA — reconstruction (diagonal averaging / Hankelization).

A grouped trajectory matrix is generally not Hankel; diagonal averaging projects it
back to a series by averaging each anti-diagonal (see report/ssa_math_notes.md):

    g_s = mean over { Y[i, j] : i + j = s }   for s = 0, ..., N-1,  N = L + K - 1.

Applied to the full (all-component) grouped matrix this returns the original series to
machine precision -- the reconstruction identity asserted by the Day-6 tests.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .decomposition import Decomposition

__all__ = [
    "diagonal_average",
    "reconstruct_series",
    "reconstruct_mssa",
]


def diagonal_average(matrix: np.ndarray) -> np.ndarray:
    """Hankelize an L x K matrix into a length-(L+K-1) series by anti-diagonal means.

    Vectorised: every entry Y[i, j] contributes to output index s = i + j; the result is
    the per-s mean. ``np.bincount`` sums values and counts per anti-diagonal in one pass
    (O(L*K)), which is far faster than a Python loop over diagonals for long series.
    """
    Y = np.asarray(matrix, dtype=float)
    if Y.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {Y.shape}")
    L, K = Y.shape
    n = L + K - 1
    s = (np.arange(L)[:, None] + np.arange(K)[None, :]).ravel()  # anti-diagonal index
    sums = np.bincount(s, weights=Y.ravel(), minlength=n)
    counts = np.bincount(s, minlength=n)  # anti-diagonal lengths (always > 0)
    return sums / counts


def _as_groups(
    groups: Sequence[Sequence[int]] | Mapping[str, Sequence[int]] | None,
    rank: int,
) -> dict[str, np.ndarray]:
    """Normalise the groups argument into an ordered {label: index-array} dict."""
    if groups is None:
        return {"all": np.arange(rank)}
    if isinstance(groups, Mapping):
        return {str(k): np.asarray(list(v), dtype=int) for k, v in groups.items()}
    return {str(i): np.asarray(list(g), dtype=int) for i, g in enumerate(groups)}


def reconstruct_series(
    decomposition: Decomposition,
    groups: Sequence[Sequence[int]] | Mapping[str, Sequence[int]] | None = None,
) -> dict[str, np.ndarray]:
    """Reconstruct univariate component series for each group of eigentriple indices.

    Parameters
    ----------
    decomposition : the eigentriple decomposition of a univariate trajectory matrix.
    groups : list of index lists, or {label: indices}. None -> a single "all" group
             (which reproduces the original series).

    Returns
    -------
    dict mapping group label -> reconstructed series (length N = L + K - 1).
    """
    g = _as_groups(groups, decomposition.rank)
    return {label: diagonal_average(decomposition.reconstruct_matrix(idx)) for label, idx in g.items()}


def reconstruct_mssa(
    decomposition: Decomposition,
    widths: Sequence[int],
    groups: Sequence[Sequence[int]] | Mapping[str, Sequence[int]] | None = None,
) -> dict[str, np.ndarray]:
    """Reconstruct per-channel component series from an MSSA decomposition.

    The grouped trajectory matrix is split column-wise into the per-channel blocks
    (given by ``widths`` from :func:`rmssa.embedding.mssa_trajectory_matrix`) and each
    block is diagonal-averaged separately.

    Returns
    -------
    dict mapping group label -> array of shape (p, N_j-aligned). When all channels
    share length N, the result for each group is a (p, N) ndarray; with unequal
    lengths it is a list of per-channel series.
    """
    g = _as_groups(groups, decomposition.rank)
    widths = list(widths)
    bounds = np.cumsum([0, *widths])
    out: dict[str, np.ndarray | list[np.ndarray]] = {}
    for label, idx in g.items():
        M = decomposition.reconstruct_matrix(idx)  # (L, K_tot)
        channel_series = [
            diagonal_average(M[:, bounds[j] : bounds[j + 1]]) for j in range(len(widths))
        ]
        lengths = {s.shape[0] for s in channel_series}
        out[label] = np.vstack(channel_series) if len(lengths) == 1 else channel_series
    return out
