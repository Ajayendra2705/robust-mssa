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
        """Fit on a single series or a multi-channel panel.

        Accepted inputs:
          * 1-D array            -> univariate SSA.
          * list/tuple of 1-D    -> MSSA, one channel per element (lengths may differ).
          * 2-D array or DataFrame -> MSSA panel in **(T, p)** layout, i.e. rows are
            time points and columns are series, matching X in R^{T x p} from the
            proposal and :func:`rmssa.datasets.make_synthetic_panel`. Each of the p
            columns becomes a channel.
        """
        if isinstance(series, (list, tuple)):
            channels = [np.asarray(c, dtype=float).ravel() for c in series]
            self._set_multivariate(channels)
        else:
            arr = _to_ndarray(series)
            if arr.ndim == 1:
                f = arr.ravel()
                self.multivariate_ = False
                self.n_channels_ = 1
                self.lengths_ = [f.shape[0]]
                self.H_ = trajectory_matrix(f, self.window)
                self.widths_ = [self.H_.shape[1]]
            elif arr.ndim == 2:
                # (T, p): columns are channels
                channels = [arr[:, j] for j in range(arr.shape[1])]
                self._set_multivariate(channels)
            else:
                raise ValueError(f"series must be 1-D or 2-D, got {arr.ndim}-D")

        self.decomposition_ = self.backend.decompose(self.H_)
        return self

    def _set_multivariate(self, channels: list[np.ndarray]) -> None:
        self.multivariate_ = True
        self.n_channels_ = len(channels)
        self.lengths_ = [c.shape[0] for c in channels]
        self.H_, self.widths_ = mssa_trajectory_matrix(channels, self.window)

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

        Univariate: the w-correlation among the group series directly. MSSA: the mean
        of the per-channel w-correlation matrices (each channel weighted by its own
        anti-diagonal weights), which keeps the diagnostic well-defined even when
        channels have different lengths.
        """
        comps = self.reconstruct(groups)
        labels = list(comps.keys())
        if not self.multivariate_:
            stack = np.vstack([np.asarray(comps[label]) for label in labels])
            return wcorrelation_matrix(stack, self.window)
        mats = []
        for j in range(self.n_channels_):
            stack = np.vstack([np.asarray(comps[label])[j] for label in labels])
            mats.append(wcorrelation_matrix(stack, self.window))
        return np.mean(mats, axis=0)

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


def _to_ndarray(series) -> np.ndarray:
    """Coerce array-likes (incl. pandas DataFrame/Series) to a float ndarray."""
    values = getattr(series, "to_numpy", None)
    if callable(values):  # pandas DataFrame / Series
        series = series.to_numpy()
    return np.asarray(series, dtype=float)
