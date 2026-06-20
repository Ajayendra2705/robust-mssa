"""End-to-end (M)SSA orchestrator.

Ties the four stages together behind one object:

    embedding -> decomposition (pluggable backend) -> grouping -> reconstruction

Handles both the univariate case (a single series) and the multivariate / horizontal
MSSA case (a list of series sharing a window L). The decomposition backend is injected,
so swapping StandardSVD for a robust backend (Phase 2) needs no change here.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .decomposition import Decomposition, DecompositionBackend, StandardSVD
from .embedding import mssa_trajectory_matrix, trajectory_matrix
from .grouping import elementary_wcorrelation, suggest_groups_by_contribution, wcorrelation_matrix
from .reconstruction import reconstruct_mssa, reconstruct_series

__all__ = ["MSSA"]


class MSSA:
    """Singular Spectrum Analysis for one or many series.

    Parameters
    ----------
    window : window length L.
    rank   : optional truncation rank passed to the default backend (ignored if an
             explicit ``backend`` is given).
    backend : a DecompositionBackend instance. Defaults to StandardSVD(rank=rank).

    Usage
    -----
    >>> model = MSSA(window=50).fit(series)            # univariate
    >>> model = MSSA(window=50).fit([s1, s2, s3])      # MSSA
    >>> comps = model.reconstruct({"trend": [0], "season": [1, 2]})
    """

    def __init__(
        self,
        window: int,
        rank: int | None = None,
        backend: DecompositionBackend | None = None,
    ):
        self.window = int(window)
        self.rank = rank
        self.backend = backend if backend is not None else StandardSVD(rank=rank)

        # populated by fit()
        self.multivariate_: bool | None = None
        self.n_channels_: int | None = None
        self.lengths_: list[int] | None = None
        self.widths_: list[int] | None = None
        self.H_: np.ndarray | None = None
        self.decomposition_: Decomposition | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, series) -> "MSSA":
        """Fit on a single series (1-D) or a list/2-D array of channels."""
        arr = np.asarray(series, dtype=float) if not isinstance(series, (list, tuple)) else None
        if isinstance(series, (list, tuple)) or (arr is not None and arr.ndim == 2):
            channels = list(series) if isinstance(series, (list, tuple)) else list(arr)
            channels = [np.asarray(c, dtype=float).ravel() for c in channels]
            self.multivariate_ = True
            self.n_channels_ = len(channels)
            self.lengths_ = [c.shape[0] for c in channels]
            self.H_, self.widths_ = mssa_trajectory_matrix(channels, self.window)
        else:
            f = arr.ravel()
            self.multivariate_ = False
            self.n_channels_ = 1
            self.lengths_ = [f.shape[0]]
            self.H_ = trajectory_matrix(f, self.window)
            self.widths_ = [self.H_.shape[1]]

        self.decomposition_ = self.backend.decompose(self.H_)
        return self

    # --------------------------------------------------------------- helpers
    def _check_fitted(self) -> Decomposition:
        if self.decomposition_ is None:
            raise RuntimeError("MSSA is not fitted yet; call fit() first.")
        return self.decomposition_

    @property
    def decomposition(self) -> Decomposition:
        return self._check_fitted()

    def contributions(self) -> np.ndarray:
        """Relative variance share per eigentriple (scree values)."""
        return self._check_fitted().contributions()

    # --------------------------------------------------------- reconstruction
    def reconstruct(
        self,
        groups: Sequence[Sequence[int]] | Mapping[str, Sequence[int]] | None = None,
    ):
        """Reconstruct component series for each group of eigentriple indices.

        Univariate -> {label: series}. MSSA -> {label: (p, N) array} (per channel).
        ``groups=None`` reproduces the original input.
        """
        d = self._check_fitted()
        if self.multivariate_:
            return reconstruct_mssa(d, self.widths_, groups)
        return reconstruct_series(d, groups)

    def reconstruct_full(self):
        """Convenience: the all-component reconstruction (== original input)."""
        return self.reconstruct(None)["all"]

    # ------------------------------------------------------------ diagnostics
    def wcorrelation(self, n_components: int | None = None) -> np.ndarray:
        """w-correlation matrix among the leading elementary components."""
        d = self._check_fitted()
        return elementary_wcorrelation(d, self.window, n_components)

    def group_wcorrelation(self, groups) -> np.ndarray:
        """w-correlation matrix between reconstructed *group* series.

        For MSSA the per-channel series are stacked across channels before correlating.
        """
        comps = self.reconstruct(groups)
        stack = []
        for v in comps.values():
            v = np.asarray(v)
            stack.append(v.ravel() if v.ndim > 1 else v)
        n = self.lengths_[0]
        # use a representative length for weights (channels share N in the common case)
        return wcorrelation_matrix(np.vstack([s[: len(s)] for s in stack]), self.window) \
            if len({len(s) for s in stack}) == 1 else _ragged_wcorr(stack, self.window, n)

    def suggest_groups(self, threshold: float = 0.99) -> dict[str, list[int]]:
        return suggest_groups_by_contribution(self._check_fitted(), threshold)

    # --------------------------------------------------------------- config
    @classmethod
    def from_config(cls, config: Mapping) -> "MSSA":
        """Build from a plain dict (e.g. parsed YAML).

        Recognised keys: ``window`` (required), ``rank`` (optional).
        """
        if "window" not in config:
            raise KeyError("config must contain 'window'")
        return cls(window=int(config["window"]), rank=config.get("rank"))


def _ragged_wcorr(stack, window, n):
    # fallback for unequal-length group series: truncate to common length
    m = min(len(s) for s in stack)
    return wcorrelation_matrix(np.vstack([s[:m] for s in stack]), window)
