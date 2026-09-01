"""Stage 2 of (M)SSA — decomposition.

The trajectory matrix H is factorised into eigentriples (sqrt(lambda_i), U_i, V_i):

    H = sum_i s_i * U_i V_i^T,    s_i = sqrt(lambda_i).

This is the *interchangeable* step of the project. Every backend implements the same
:class:`DecompositionBackend` contract and returns a :class:`Decomposition`, so that
embedding / grouping / reconstruction / forecasting never depend on *which* backend
produced the factorisation.

Backends:
  * :class:`StandardSVD`      — classical L2 SVD (baseline).
  * :class:`RobRSVD`          — RHSSA: Huber-weighted robust SVD (Zhang et al. 2013).
  * :class:`AlternatingL1SVD` — RLSSA: L1-norm robust SVD (Hawkins et al. 2001).

The two robust backends are the two robust SSA model-fit algorithms of Rodrigues,
Pimentel, Messala & Kazemi (2020, *Entropy* 22(1):8), per the supervisor's 24 Jul 2026
directive. They share one IRLS engine and are added in Phase 2 with no change to
downstream code (the modularity claim).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np

__all__ = [
    "Decomposition",
    "DecompositionBackend",
    "StandardSVD",
    "RobustSVD",
    "RobRSVD",
    "AlternatingL1SVD",
]


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


# ============================================================================
# Robust SVD backends (Phase 2, Days 13/15)
# ----------------------------------------------------------------------------
# Supervisor directive (24 Jul 2026): replace the L2 SVD with the *two* robust
# SVD algorithms from the attached paper, then compare classical vs robust and
# univariate vs multivariate. Both are implemented here behind the *same*
# ``decompose`` contract, so mssa.py / grouping / reconstruction / forecasting
# never learn which backend produced the factorisation (the modularity claim).
#
# Both share one engine: robust rank-r fitting by **iteratively reweighted
# imputation** (IRLS). Given the current rank-r model L_r, residuals R = H - L_r
# get a per-cell trust weight W = psi(R)/R in [0, 1] (down-weighting large
# residuals); down-weighted cells are pulled toward the model,
#     Z = W .* H + (1 - W) .* L_r,
# and the model is refreshed as the truncated L2 SVD of the cleaned matrix Z.
# Iterate to convergence. The weight is estimated against the *full* rank-r model
# (not a single rank-1 layer), so legitimate lower-rank signal is NOT mistaken
# for outliers -- the flaw of naive rank-1 deflation.
#
# Reduction to the ordinary SVD at epsilon = 0 holds only when r >= rank(signal).
# Then R -> 0, W -> 1, Z -> H and the two backends agree with StandardSVD to ~1e-8,
# so the "classical vs robust" comparison is fair. Below the signal rank the residual
# is not noise but *discarded signal*, the MAD scale is set by it, and the Huber weight
# down-weights legitimate structure: on a noise-free rank-6 panel the robust and
# classical rank-3 subspaces sit 0.57 apart with no contamination whatever. The
# approximation error is barely affected (within 1-5% of the L2 optimum at every rank),
# so signal-recovery comparisons stay fair -- but *subspace* comparisons below the
# signal rank do not. See report/results_robust_init.md.
#
# The two algorithms are the two robust SSA model-fit variants of
#   Rodrigues, Pimentel, Messala & Kazemi (2020), "The Decomposition and
#   Forecasting of Mutual Investment Funds Using SSA", Entropy 22(1):8.
# They differ only in the M-estimator weight W(r):
#
#   * RobRSVD           -> RHSSA: Huber weights, w = min(1, c*scale/|r|),
#                          c = 1.345 (a special case of robust regularized SVD;
#                          Zhang, Shen & Huang 2013; R RobRSVD(rough=TRUE,
#                          uspar=0, vspar=0)).
#   * AlternatingL1SVD  -> RLSSA: L1 / least-absolute-deviations weights,
#                          w = min(1, scale/|r|) (Hawkins, Liu & Young 2001;
#                          R robustSVD() in pcaMethods) -- more aggressive tail
#                          down-weighting (higher breakdown).
#
# Losses and the Huber constant match the paper exactly. The *solver* here is a
# joint IRLS-by-imputation, not the R packages' per-component deflation, so a
# numerical cross-check vs pcaMethods::robustSVD / RobRSVD is a TODO for direct
# comparability (Day 14/16). The shared engine makes swapping the weight function
# (hence the algorithm) a one-line change.
# ============================================================================


def _winsorized(H: np.ndarray, k: float = 2.5) -> np.ndarray:
    """Column-wise Winsorization of H at ``median +- k * MAD`` — an outlier-resistant
    stand-in for H, used only to *initialise* the IRLS iteration.

    Why this exists (measured, see ``report/results_robust_init.md``): initialising at
    the classical SVD of H starts the iteration inside the basin the outliers have
    already rotated it into, and IRLS-by-imputation cannot leave it — down-weighted
    cells are replaced by the *current* model's own values, so the current model is a
    fixed point. Starting from a Winsorized copy instead does not change the estimator,
    only which fixed point it reaches.
    """
    med = np.median(H, axis=0, keepdims=True)
    scale = np.maximum(1.4826 * np.median(np.abs(H - med), axis=0, keepdims=True), 1e-12)
    return np.clip(H, med - k * scale, med + k * scale)


def _mad_scale(r: np.ndarray, eps: float = 1e-12) -> float:
    """Robust scale estimate: normalised median-absolute-deviation of residuals.

    ``1.4826 * MAD`` is a consistent estimator of sigma for Gaussian data. Floored
    at ``eps`` so a (near-)perfect fit does not divide by zero.
    """
    r = r.ravel()
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    return max(1.4826 * mad, eps)


def _huber_weights(r: np.ndarray, scale: float, c: float) -> np.ndarray:
    """Huber trust weights w = min(1, c*scale/|r|): full weight inside c*scale, then c*scale/|r|."""
    z = np.abs(r) / scale
    w = np.ones_like(z)
    big = z > c
    w[big] = c / z[big]
    return w


def _l1_weights(r: np.ndarray, scale: float, c: float) -> np.ndarray:
    """L1 / LAD trust weights w = min(1, scale/|r|) (IRLS influence 1/|r|, clipped). ``c`` unused."""
    z = np.abs(r) / scale
    w = np.ones_like(z)
    big = z > 1.0
    w[big] = 1.0 / z[big]
    return w


class RobustSVD(DecompositionBackend):
    """Robust SVD by iteratively reweighted imputation (IRLS) of the rank-r model.

    Concrete backends (:class:`RobRSVD`, :class:`AlternatingL1SVD`) only choose the
    residual weight function. On clean data *at or above the signal rank* this converges
    to the ordinary SVD (weights -> 1), so robust backends agree with
    :class:`StandardSVD` at contamination epsilon = 0; a planted outlier is
    *down-weighted* rather than allowed to rotate the leading subspace. Because the
    weight is estimated against the full rank-r model, genuine lower-rank components are
    not mistaken for outliers. Below the signal rank that equivalence fails — see the
    module comment. The returned ``U``/``Vt`` are orthonormal (final truncated SVD).

    Requires an explicit ``rank`` (r): the robust low-rank target is rank-r, so the
    truncation is part of the estimator, not a post-hoc cut. ``rank=None`` falls
    back to the full numerical rank (equivalent to standard SVD on clean data).

    Parameters
    ----------
    rank : target rank r of the robust low-rank fit.
    c : tuning constant for the weight function (Huber default 1.345).
    max_iter : max IRLS sweeps.
    tol : convergence tolerance on the relative change of the rank-r model.
    init : ``"robust"`` (default) starts from the SVD of a Winsorized copy of H;
           ``"classical"`` starts from the SVD of H itself, which is what Phase 2 did.
           The two agree exactly unless the classical start is already captured by the
           outliers — see :func:`_winsorized`.

    Attributes (set after ``decompose``)
    ------------------------------------
    n_iter_ : number of IRLS sweeps actually run.
    converged_ : whether the tolerance was met before ``max_iter``.
    init_used_ : which starting point produced the returned fit ("classical"/"robust").
    """

    #: subclasses set this; it is the only thing that differs between algorithms
    _weight_fn = staticmethod(_huber_weights)
    _default_c = 1.345

    def __init__(
        self,
        rank: int | None = None,
        c: float | None = None,
        max_iter: int = 200,
        tol: float = 1e-9,
        init: str = "auto",
    ):
        if rank is not None and rank < 1:
            raise ValueError(f"rank must be >= 1 or None, got {rank}")
        if init not in ("auto", "robust", "classical"):
            raise ValueError(f"init must be 'auto', 'robust' or 'classical', got {init!r}")
        self.rank = rank
        self.c = self._default_c if c is None else float(c)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.init = init

    @staticmethod
    def _truncated_svd(M: np.ndarray, r: int):
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        return U[:, :r], s[:r], Vt[:r]

    def _rho(self, R: np.ndarray, scale: float) -> float:
        """The M-estimation objective this backend's weights are the IRLS weights of.

        ``rho(z) = z^2/2`` inside ``c``, ``c(|z| - c/2)`` outside — Huber at c = 1.345 for
        RHSSA, and the same function at c = 1 for the L1/LAD weights of RLSSA. Used only
        to choose between the fits produced by different starting points, so it must be
        evaluated at a *common* scale: each fit's own MAD scale would let a fit that has
        absorbed the outliers shrink its scale and win.
        """
        z = np.abs(R) / scale
        return float(np.sum(np.where(z <= self.c, 0.5 * z ** 2, self.c * (z - 0.5 * self.c))))

    def _irls(self, H: np.ndarray, start: np.ndarray, r: int):
        """IRLS-by-imputation sweeps from a given rank-r starting model."""
        L_r = start
        prev_norm = np.linalg.norm(L_r) or 1.0
        U, s, Vt = self._truncated_svd(L_r, r)
        n_iter, converged = 0, False

        for it in range(self.max_iter):
            R = H - L_r
            scale = _mad_scale(R)
            W = self._weight_fn(R, scale, self.c)      # per-cell trust in [0, 1]
            Z = W * H + (1.0 - W) * L_r                # impute down-weighted cells
            U, s, Vt = self._truncated_svd(Z, r)
            L_new = (U * s) @ Vt

            change = np.linalg.norm(L_new - L_r) / prev_norm
            L_r = L_new
            prev_norm = np.linalg.norm(L_r) or 1.0
            n_iter = it + 1
            if change <= self.tol:
                converged = True
                break
        return L_r, U, s, Vt, n_iter, converged

    # --------------------------------------------------------------- backend
    def decompose(self, H: np.ndarray) -> Decomposition:
        H = np.asarray(H, dtype=float)
        if H.ndim != 2:
            raise ValueError(f"H must be 2-D, got shape {H.shape}")

        max_rank = min(H.shape)
        r = max_rank if self.rank is None else min(self.rank, max_rank)

        # The iteration is a fixed-point scheme, not a descent method, so the starting
        # point decides which fixed point it reaches. Neither candidate start is safe
        # alone: the classical SVD of H is already rotated onto the outliers whenever
        # they dominate (and IRLS-by-imputation cannot leave that basin), while the
        # Winsorized start perturbs an exactly-low-rank clean matrix off its exact
        # solution and sticks there. So run both and keep whichever fit actually has the
        # lower objective, scored at one common scale. On clean data the classical fit
        # scores exactly 0 and wins; under heavy contamination the Winsorized one wins.
        starts = {"classical": H, "robust": _winsorized(H)}
        if self.init != "auto":
            starts = {self.init: starts[self.init]}

        fits = {}
        for name, seed_matrix in starts.items():
            Us, ss, Vts = self._truncated_svd(seed_matrix, r)
            fits[name] = self._irls(H, (Us * ss) @ Vts, r)

        if len(fits) == 1:
            self.init_used_ = next(iter(fits))
        else:
            # common scale: fixed by the Winsorized fit, so it does not depend on which
            # candidate is being scored. Ties (both residuals ~0) keep "classical".
            sigma = _mad_scale(H - fits["robust"][0])
            self.init_used_ = (
                "robust"
                if self._rho(H - fits["robust"][0], sigma) < self._rho(H - fits["classical"][0], sigma)
                else "classical"
            )

        L_r, U, s, Vt, self.n_iter_, self.converged_ = fits[self.init_used_]

        # drop any numerically-zero trailing singular values (keep >= 1)
        if s.size:
            cutoff = 1e-12 * s[0] * max(H.shape)
            keep = max(int(np.sum(s > cutoff)), 1)
        else:
            keep = 1
            U = np.zeros((H.shape[0], 1))
            s = np.zeros(1)
            Vt = np.zeros((1, H.shape[1]))

        # sign convention: largest-|.| entry of each left vector positive
        for i in range(keep):
            k = int(np.argmax(np.abs(U[:, i])))
            if U[k, i] < 0:
                U[:, i] = -U[:, i]
                Vt[i] = -Vt[i]

        return Decomposition(U=U[:, :keep].copy(), s=s[:keep].copy(), Vt=Vt[:keep].copy())


class RobRSVD(RobustSVD):
    """RHSSA — robust SVD with Huber M-estimator weights (Rodrigues et al. 2020).

    The Huber robust SVD is a special case of the robust *regularized* SVD of Zhang,
    Shen & Huang (2013); the roughness penalties on the singular vectors are deferred
    (defaults off, matching the paper's ``RobRSVD(rough=TRUE, uspar=0, vspar=0)``),
    leaving the plain Huber robust SVD. Down-weights outliers smoothly: full weight
    within ``c*scale``, ~c*scale/|r| beyond. Default ``c = 1.345`` as in the paper.
    """

    _weight_fn = staticmethod(_huber_weights)
    _default_c = 1.345


class AlternatingL1SVD(RobustSVD):
    """RLSSA — L1-norm robust SVD (Hawkins, Liu & Young 2001; Rodrigues et al. 2020).

    Robustifies the SVD with the L1 / least-absolute-deviations loss via IRLS trust
    weights ``w = min(1, scale/|r|)`` — the LAD analogue of the L2 SVD. Higher
    breakdown than Huber against gross outliers, at the cost of slightly slower
    convergence. Reference implementation: ``robustSVD()`` in the R package *pcaMethods*.
    """

    _weight_fn = staticmethod(_l1_weights)
    _default_c = 1.0  # unused by L1 weights; kept for a uniform constructor
