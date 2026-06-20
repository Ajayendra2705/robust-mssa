"""Stage 2 of (M)SSA — decomposition.

The trajectory matrix H is factorised into eigentriples (sqrt(lambda_i), U_i, V_i):

    H = sum_i s_i * U_i V_i^T,    s_i = sqrt(lambda_i).

This is the *interchangeable* step of the project. Every backend implements the same
:class:`DecompositionBackend` contract and returns a :class:`Decomposition`, so that
embedding / grouping / reconstruction / forecasting never depend on *which* backend
produced the factorisation. Robust backends (column-weighted IRLS SVD, kernel robust
SVD, ...) are added in Phase 2 with no change to downstream code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np

__all__ = ["Decomposition", "DecompositionBackend", "StandardSVD"]


@dataclass(frozen=True)
class Decomposition:
    """An eigentriple decomposition of a trajectory matrix.

    Attributes
    ----------
    U  : ndarray (L, d) left singular vectors (orthonormal columns).
    s  : ndarray (d,)   singular values s_i = sqrt(lambda_i), descending, > 0.
    Vt : ndarray (d, K) right singular vectors as rows (orthonormal rows).
    """

    U: np.ndarray
    s: np.ndarray
    Vt: np.ndarray

    def __post_init__(self) -> None:
        if self.U.ndim != 2 or self.Vt.ndim != 2 or self.s.ndim != 1:
            raise ValueError("U, Vt must be 2-D and s 1-D")
        d = self.s.shape[0]
        if self.U.shape[1] != d or self.Vt.shape[0] != d:
            raise ValueError(
                f"inconsistent ranks: U has {self.U.shape[1]}, s has {d}, Vt has {self.Vt.shape[0]}"
            )

    @property
    def rank(self) -> int:
        return self.s.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """Shape (L, K) of the matrix this decomposition reconstructs."""
        return self.U.shape[0], self.Vt.shape[1]

    def contributions(self) -> np.ndarray:
        """Relative variance share lambda_i / sum_j lambda_j (the scree values)."""
        lam = self.s ** 2
        total = lam.sum()
        if total == 0:
            return np.zeros_like(lam)
        return lam / total

    def elementary(self, i: int) -> np.ndarray:
        """The i-th rank-1 elementary matrix X_i = s_i * U_i V_i^T."""
        return self.s[i] * np.outer(self.U[:, i], self.Vt[i])

    def reconstruct_matrix(self, indices: Iterable[int] | None = None) -> np.ndarray:
        """Sum of elementary matrices over ``indices`` (default: all).

        Vectorised: (U[:, idx] * s[idx]) @ Vt[idx].  With all indices this returns
        the original trajectory matrix to machine precision.
        """
        if indices is None:
            idx = np.arange(self.rank)
        else:
            idx = np.fromiter(indices, dtype=int)
        return (self.U[:, idx] * self.s[idx]) @ self.Vt[idx]


class DecompositionBackend(ABC):
    """Interface every decomposition method must implement.

    The single contract: map a trajectory matrix H (L x K) to a
    :class:`Decomposition`. Truncation rank, if any, is backend state.
    """

    @abstractmethod
    def decompose(self, H: np.ndarray) -> Decomposition:  # pragma: no cover - abstract
        ...

    def __call__(self, H: np.ndarray) -> Decomposition:
        return self.decompose(H)


class StandardSVD(DecompositionBackend):
    """Classic (non-robust) SVD backend — the baseline MSSA decomposition.

    Solves the L2 / Eckart-Young problem ``min_{rank(M)<=r} ||H - M||_F``.
    This is the estimator Robust MSSA will replace; it is the reference against
    which robust backends must agree on clean (uncontaminated) data.

    Parameters
    ----------
    rank : optional truncation rank r. If None, keep all singular triples with a
           strictly positive singular value (numerical rank).
    tol  : singular values <= tol * s_max * max(L, K) are treated as zero
           (relative threshold for numerical rank).
    """

    def __init__(self, rank: int | None = None, tol: float = 1e-12):
        if rank is not None and rank < 1:
            raise ValueError(f"rank must be >= 1 or None, got {rank}")
        self.rank = rank
        self.tol = float(tol)

    def decompose(self, H: np.ndarray) -> Decomposition:
        H = np.asarray(H, dtype=float)
        if H.ndim != 2:
            raise ValueError(f"H must be 2-D, got shape {H.shape}")
        U, s, Vt = np.linalg.svd(H, full_matrices=False)

        # Numerical-rank cutoff: drop ~zero singular values.
        if s.size:
            cutoff = self.tol * s[0] * max(H.shape)
            keep = int(np.sum(s > cutoff))
        else:
            keep = 0
        keep = max(keep, 1) if s.size else 0

        if self.rank is not None:
            keep = min(keep, self.rank)

        return Decomposition(U=U[:, :keep].copy(), s=s[:keep].copy(), Vt=Vt[:keep].copy())
