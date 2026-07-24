"""Ground-truth & stability metrics (Day 12).

Metrics for the synthetic-validation and empirical phases, in three groups:

  * **Subspace recovery** — how close is an *estimated* factor subspace to the
    *true* one? Measured by principal angles between the two column spans, so the
    metric is invariant to the arbitrary within-subspace rotation / sign / ordering
    of singular vectors. This is the primary correctness metric on synthetic data
    where the true low-rank factor subspace is known.
  * **Reconstruction error** — RMSE / MAE of an estimate against a clean target
    (e.g. reconstructed signal vs the known clean signal S), plus a scale-free
    relative-Frobenius variant for cross-experiment aggregation.
  * **Factor stability** — how much does the leading factor subspace move between
    two fits (adjacent rolling windows, bootstrap resamples, robust variants)?
    Same principal-angle machinery, applied to two *estimated* subspaces.

All subspace functions accept factor bases as columns of a matrix ``U`` (L x r);
non-orthonormal inputs are orthonormalised internally, so raw ``Decomposition.U``
or a stack of factor series can be passed directly.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "principal_angles",
    "subspace_distance",
    "grassmann_distance",
    "subspace_overlap",
    "rmse",
    "mae",
    "relative_frobenius",
    "signal_recovery_error",
    "factor_stability",
]


# --------------------------------------------------------------------- subspace
def _orthonormal_basis(U: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Orthonormal basis (L x k) for the column span of U, dropping null directions."""
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[:, None]
    if U.ndim != 2:
        raise ValueError(f"basis must be 1-D or 2-D, got shape {U.shape}")
    if U.shape[1] == 0:
        return U.reshape(U.shape[0], 0)
    # QR gives an orthonormal span; SVD-based rank check drops degenerate columns.
    Q, R = np.linalg.qr(U)
    diag = np.abs(np.diag(R))
    if diag.size:
        keep = diag > rtol * diag.max()
        Q = Q[:, keep]
    return Q


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between the column spans of A and B.

    Computed from the singular values (canonical correlations) of ``Qa^T Qb`` where
    ``Qa``, ``Qb`` are orthonormal bases. Returns ``min(dim A, dim B)`` angles in
    ``[0, pi/2]``: 0 == a shared direction, pi/2 == an orthogonal direction.
    """
    Qa = _orthonormal_basis(A)
    Qb = _orthonormal_basis(B)
    if Qa.shape[1] == 0 or Qb.shape[1] == 0:
        return np.array([np.pi / 2])
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)
    return np.arccos(sv)  # ascending, since svd values are descending


def subspace_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Sine of the *largest* principal angle between spans of A and B (the gap).

    In ``[0, 1]``: 0 == one span contains the other (identical if equal dim),
    1 == the spans share an orthogonal direction. This is the standard "does the
    estimated factor subspace recover the true one?" error on synthetic data.
    """
    theta = principal_angles(A, B)
    return float(np.sin(theta.max()))


def grassmann_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Grassmann (geodesic) distance ``||theta||_2`` between equal-dimension spans.

    Uses all principal angles, so it is more sensitive than :func:`subspace_distance`
    (which uses only the worst one). Defined for subspaces of the same dimension.
    """
    return float(np.linalg.norm(principal_angles(A, B)))


def subspace_overlap(A: np.ndarray, B: np.ndarray) -> float:
    """Mean squared canonical correlation ``mean(cos^2 theta_i)`` in ``[0, 1]``.

    1 == identical subspace, 0 == orthogonal. A convenient *similarity* (rather than
    distance) for stability tables; equals ``1 - mean(sin^2 theta)``.
    """
    theta = principal_angles(A, B)
    return float(np.mean(np.cos(theta) ** 2))


# ---------------------------------------------------------------- reconstruction
def _align(estimate: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    est = np.asarray(estimate, dtype=float)
    tgt = np.asarray(target, dtype=float)
    if est.shape != tgt.shape:
        raise ValueError(f"shape mismatch: estimate {est.shape} vs target {tgt.shape}")
    return est, tgt


def rmse(estimate: np.ndarray, target: np.ndarray) -> float:
    """Root-mean-square error between an estimate and a target (same shape)."""
    est, tgt = _align(estimate, target)
    return float(np.sqrt(np.mean((est - tgt) ** 2)))


def mae(estimate: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error between an estimate and a target (same shape)."""
    est, tgt = _align(estimate, target)
    return float(np.mean(np.abs(est - tgt)))


def relative_frobenius(estimate: np.ndarray, target: np.ndarray) -> float:
    """Scale-free error ``||estimate - target||_F / ||target||_F``.

    Comparable across panels of different scale, so it is the default headline
    error for aggregating the contamination sweep.
    """
    est, tgt = _align(estimate, target)
    denom = np.linalg.norm(tgt)
    return float(np.linalg.norm(est - tgt) / denom) if denom > 0 else float(np.linalg.norm(est - tgt))


def signal_recovery_error(estimate: np.ndarray, clean_signal: np.ndarray) -> float:
    """Relative-Frobenius error of a reconstruction against the known clean signal S.

    The core synthetic-data question: fitting on contaminated ``X = S + N + O``, how
    well is the clean low-rank signal ``S`` recovered? Lower is better; robust methods
    should beat classical ones as contamination rises.
    """
    return relative_frobenius(estimate, clean_signal)


# -------------------------------------------------------------------- stability
def factor_stability(U_a: np.ndarray, U_b: np.ndarray) -> float:
    """Stability of the leading factor subspace between two fits, in ``[0, 1]``.

    ``= subspace_overlap`` (mean squared canonical correlation): 1 == the two fits
    span the same leading-factor subspace, 0 == unrelated. Used for rolling-window
    stability (adjacent windows), bootstrap stability, and cross-variant agreement.
    Invariant to sign flips, permutations and rotations of the factors.
    """
    return subspace_overlap(U_a, U_b)
