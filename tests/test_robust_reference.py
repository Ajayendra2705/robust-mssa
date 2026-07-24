"""Day-14 cross-check: the robust backends against an INDEPENDENT optimiser.

The production backends solve the robust low-rank problem by IRLS-by-imputation
(reweight -> impute down-weighted cells -> re-SVD). Here we solve the *same*
M-estimator objective with a structurally different optimiser -- **robust weighted
alternating least squares (ALS)** that never forms an imputed matrix and never calls
a full SVD in the loop, updating U and V by weighted normal equations instead.

Two independent optimisers landing on the same factor subspace is strong evidence
the estimator (not just one implementation) is correct. This is the runnable stand-in
for the authoritative check against the paper's R packages
(``pcaMethods::robustSVD``, ``RobRSVD``); that package-level check is scripted under
``experiments/02_synthetic_validation/rcheck/`` and runs when R is installed.
"""

import numpy as np
import pytest

from rmssa.decomposition import RobRSVD, AlternatingL1SVD, _huber_weights, _l1_weights, _mad_scale
from rmssa.embedding import mssa_trajectory_matrix
from rmssa.datasets import make_synthetic_panel
from rmssa.metrics import subspace_distance


def weighted_als_robust(H, r, weight_fn, c, max_iter=500, tol=1e-10):
    """Independent robust rank-r fit by reweighted alternating least squares.

    U (L x r), V (K x r) minimise sum_ij W_ij (H - U V^T)_ij^2 with W reweighted from
    the full-model residual each sweep -- the same objective as the backends, a
    different optimiser (weighted normal equations, no imputation, no in-loop SVD).
    Returns an orthonormal basis for the fitted left factor subspace.
    """
    H = np.asarray(H, float)
    L, K = H.shape
    # init from a plain SVD so both methods start from the same clean-data optimum
    U0, s0, V0t = np.linalg.svd(H, full_matrices=False)
    U = U0[:, :r] * np.sqrt(s0[:r])
    V = (V0t[:r].T) * np.sqrt(s0[:r])

    prev = np.inf
    for _ in range(max_iter):
        R = H - U @ V.T
        scale = _mad_scale(R)
        W = weight_fn(R, scale, c)
        # update V rows: V_j = (U^T diag(W_:j) U)^{-1} U^T diag(W_:j) H_:j
        for j in range(K):
            w = W[:, j]
            A = (U * w[:, None]).T @ U
            b = (U * w[:, None]).T @ H[:, j]
            V[j] = np.linalg.solve(A + 1e-12 * np.eye(r), b)
        # update U rows: U_i = (V^T diag(W_i:) V)^{-1} V^T diag(W_i:) H_i:
        for i in range(L):
            w = W[i, :]
            A = (V * w[:, None]).T @ V
            b = (V * w[:, None]).T @ H[i, :]
            U[i] = np.linalg.solve(A + 1e-12 * np.eye(r), b)

        fit = np.linalg.norm(U @ V.T)
        if abs(fit - prev) <= tol * (prev + tol):
            break
        prev = fit

    Q, _ = np.linalg.qr(U)
    return Q[:, :r]


@pytest.mark.parametrize(
    "Backend, weight_fn, c",
    [
        (RobRSVD, _huber_weights, 1.345),
        (AlternatingL1SVD, _l1_weights, 1.0),
    ],
)
def test_backend_agrees_with_independent_als(Backend, weight_fn, c):
    # Contaminated multivariate panel: the two optimisers should recover the same
    # robust leading-factor subspace.
    sp = make_synthetic_panel(T=200, p=5, k=2, noise_sd=0.03,
                              contamination=0.05, outlier_scale=8.0, seed=3)
    channels = [sp.X[:, j] for j in range(sp.X.shape[1])]
    H, _ = mssa_trajectory_matrix(channels, 40)
    r = 2

    U_backend = Backend(rank=r).decompose(H).U
    U_ref = weighted_als_robust(H, r, weight_fn, c)

    # same robust subspace up to rotation/sign (small tol: different optimisers)
    assert subspace_distance(U_backend[:, :r], U_ref) < 5e-2


def test_both_backends_beat_standard_under_contamination():
    # Sanity: on the SAME contaminated panel, both robust backends recover the clean
    # trajectory better than the classical SVD (the whole point of the exercise).
    from rmssa.decomposition import StandardSVD

    sp = make_synthetic_panel(T=250, p=6, k=2, noise_sd=0.03,
                              contamination=0.06, outlier_scale=9.0, seed=11)
    chans = [sp.X[:, j] for j in range(sp.X.shape[1])]
    chans_clean = [sp.signal[:, j] for j in range(sp.X.shape[1])]
    H, _ = mssa_trajectory_matrix(chans, 50)
    H_clean, _ = mssa_trajectory_matrix(chans_clean, 50)
    r = 2

    err_std = np.linalg.norm(
        StandardSVD(rank=r).decompose(H).reconstruct_matrix() - H_clean
    )
    for Backend in (RobRSVD, AlternatingL1SVD):
        err_rob = np.linalg.norm(
            Backend(rank=r).decompose(H).reconstruct_matrix() - H_clean
        )
        assert err_rob < err_std
