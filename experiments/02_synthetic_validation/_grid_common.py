"""Shared pieces for the synthetic-validation experiments (Days 16-18).

Defines the method grid, the fit-and-recover helpers, and the two ground-truth
scores used across the grid (Day 16) and the contamination sweep (Day 17):

  * signal-recovery error   -- ||recovered signal - clean S||_F / ||S||_F;
  * subspace-recovery error -- sin of the largest principal angle between the
    estimated leading left-factor subspace and the TRUE one (the clean-signal
    trajectory subspace). For univariate mode it is averaged over channels.

Kept out of ``src/rmssa`` on purpose: this is experiment glue, not library API.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rmssa.mssa import MSSA  # noqa: E402
from rmssa.embedding import trajectory_matrix, mssa_trajectory_matrix  # noqa: E402
from rmssa.decomposition import StandardSVD, RobRSVD, AlternatingL1SVD  # noqa: E402
from rmssa.metrics import signal_recovery_error, subspace_distance  # noqa: E402

MODES = ["univariate", "multivariate"]


def make_backends(max_iter: int, tol: float):
    """method label -> backend factory (takes rank r). Robust caps set per-experiment."""
    return {
        "classical": lambda r: StandardSVD(rank=r),
        "RHSSA_huber": lambda r: RobRSVD(rank=r, max_iter=max_iter, tol=tol),
        "RLSSA_l1": lambda r: AlternatingL1SVD(rank=r, max_iter=max_iter, tol=tol),
    }


def recover_signal(X: np.ndarray, backend_factory, L: int, r: int, mode: str) -> np.ndarray:
    """Return the (T, p) reconstructed signal for one (method, mode) config."""
    p = X.shape[1]
    if mode == "multivariate":
        channels = [X[:, j] for j in range(p)]
        model = MSSA(window=L, backend=backend_factory(r)).fit(channels)
        return np.asarray(model.reconstruct_full()).T  # (p, T) -> (T, p)
    cols = [MSSA(window=L, backend=backend_factory(r)).fit(X[:, j]).reconstruct_full()
            for j in range(p)]
    return np.column_stack(cols)  # (T, p)


def _true_left_subspace_multi(clean_signal: np.ndarray, L: int, r: int) -> np.ndarray:
    """Leading-r left subspace of the CLEAN block-Hankel trajectory (ground truth)."""
    channels = [clean_signal[:, j] for j in range(clean_signal.shape[1])]
    H, _ = mssa_trajectory_matrix(channels, L)
    return StandardSVD(rank=r).decompose(H).U


def subspace_error(X: np.ndarray, clean_signal: np.ndarray, backend_factory,
                   L: int, r: int, mode: str) -> float:
    """Principal-angle error between the estimated and true leading factor subspace.

    Multivariate: one subspace_distance on the block-Hankel. Univariate: mean over
    channels of the per-series subspace_distance (each series has its own trajectory
    subspace). Lower = better recovery of the true factor geometry.
    """
    p = X.shape[1]
    if mode == "multivariate":
        U_true = _true_left_subspace_multi(clean_signal, L, r)
        channels = [X[:, j] for j in range(p)]
        H, _ = mssa_trajectory_matrix(channels, L)
        U_est = backend_factory(r).decompose(H).U
        return subspace_distance(U_est[:, :r], U_true[:, :r])
    # univariate: average over channels
    dists = []
    for j in range(p):
        U_true = StandardSVD(rank=r).decompose(trajectory_matrix(clean_signal[:, j], L)).U
        U_est = backend_factory(r).decompose(trajectory_matrix(X[:, j], L)).U
        dists.append(subspace_distance(U_est[:, :r], U_true[:, :r]))
    return float(np.mean(dists))


def evaluate(X: np.ndarray, clean_signal: np.ndarray, backend_factory,
             L: int, r: int, mode: str) -> tuple[float, float]:
    """Fit ONCE and return (recovery_error, subspace_error) for one config.

    Avoids the double fit of calling recover_signal + subspace_error separately --
    both metrics come from the same decomposition, halving the sweep's cost.
    """
    p = X.shape[1]
    if mode == "multivariate":
        channels = [X[:, j] for j in range(p)]
        model = MSSA(window=L, backend=backend_factory(r)).fit(channels)
        rec = np.asarray(model.reconstruct_full()).T          # (T, p)
        rec_err = signal_recovery_error(rec, clean_signal)
        U_true = _true_left_subspace_multi(clean_signal, L, r)
        sub_err = subspace_distance(model.decomposition.U[:, :r], U_true[:, :r])
        return rec_err, sub_err

    # univariate: one fit per channel, both metrics per channel
    cols, sub = [], []
    for j in range(p):
        model = MSSA(window=L, backend=backend_factory(r)).fit(X[:, j])
        cols.append(model.reconstruct_full())
        U_true = StandardSVD(rank=r).decompose(trajectory_matrix(clean_signal[:, j], L)).U
        sub.append(subspace_distance(model.decomposition.U[:, :r], U_true[:, :r]))
    rec_err = signal_recovery_error(np.column_stack(cols), clean_signal)
    return rec_err, float(np.mean(sub))
