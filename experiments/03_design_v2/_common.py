"""Shared harness for the design-v2 experiments (post 25 Jul 2026 supervisor note).

Differs from ``02_synthetic_validation/_grid_common.py`` in one substantive way: the
truncation rank is no longer a single number shared by both modes. Under the new
``independent`` dependence level that would be a rigged comparison —

    p independent series each carrying an SSA-rank-``q`` signal span up to ``p*q``
    dimensions jointly, but horizontal MSSA has only ONE L-dimensional row space to
    represent all of them. Forcing MSSA to use the univariate ``r`` hands it a
    subspace too small to hold the signal it is being scored on.

So each (mode, dependence) cell gets the rank the *clean* signal actually needs, read
off the clean trajectory matrix by variance share (:func:`oracle_rank`). Both the naive
matched-rank and the capacity-matched convention are reported, because which one the
supervisor's H1 is about is exactly the open question.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rmssa.decomposition import AlternatingL1SVD, RobRSVD, StandardSVD  # noqa: E402
from rmssa.embedding import mssa_trajectory_matrix, trajectory_matrix  # noqa: E402
from rmssa.metrics import signal_recovery_error, subspace_distance  # noqa: E402
from rmssa.mssa import MSSA  # noqa: E402

MODES = ("univariate", "multivariate")

#: robust solver caps — the sweeps run thousands of fits, so tol is loosened from the
#: library default (1e-9) exactly as in Phase 2, keeping the two studies comparable
ROBUST_MAX_ITER = 60
ROBUST_TOL = 1e-6


def make_backends(max_iter: int = ROBUST_MAX_ITER, tol: float = ROBUST_TOL):
    """label -> (rank -> backend). Huber is primary; L1 is the appendix algorithm."""
    return {
        "classical": lambda r: StandardSVD(rank=r),
        "RHSSA_huber": lambda r: RobRSVD(rank=r, max_iter=max_iter, tol=tol),
        "RLSSA_l1": lambda r: AlternatingL1SVD(rank=r, max_iter=max_iter, tol=tol),
    }


#: variance share defining the "rank the clean signal occupies".
#:
#: This works only for an *exactly* low-rank signal, i.e. the synthetic base, where the
#: singular spectrum has a genuine cliff and any threshold in the gap returns the same
#: algebraic rank. For a real base there is no cliff and the rule degenerates: at 0.999
#: AirPassengers wants r=45 out of a possible L=48, and a rank-45 model in a
#: 48-dimensional space fits the outliers exactly, so classical and robust are equally
#: bad (recovery error ~1.8) and the comparison measures nothing. Even 0.99 lands at
#: r=16-23, where the robust gain has already decayed from 5.4x (r=8) to 1.3x (r=16).
#:
#: So rank is treated as an explicit design FACTOR for real bases (swept in experiment
#: 2) rather than pretending one automatic rule covers both kinds of signal.
VAR_SHARE = 0.999

#: hard ceiling on r as a fraction of L, so no cell can degenerate into "rank ~ L"
RANK_CAP_FRACTION = 0.5


def oracle_rank(H: np.ndarray, var_share: float = VAR_SHARE, cap: int | None = None) -> int:
    """Smallest rank explaining ``var_share`` of the CLEAN trajectory matrix's variance.

    Read off the clean signal, so it is the rank the signal genuinely occupies rather
    than a tuning knob. For an exactly low-rank signal (the synthetic base) this returns
    the algebraic rank; for a real base series it returns the effective rank.
    """
    s = np.linalg.svd(np.asarray(H, dtype=float), compute_uv=False)
    lam = s ** 2
    total = lam.sum()
    if total <= 0:
        return 1
    r = int(np.searchsorted(np.cumsum(lam) / total, var_share) + 1)
    limit = min(H.shape) if cap is None else min(cap, min(H.shape))
    return int(np.clip(r, 1, max(1, limit - 1)))


def clean_ranks(
    signal: np.ndarray, L: int, var_share: float = VAR_SHARE
) -> tuple[int, int]:
    """``(r_univariate, r_multivariate)`` needed by the clean signal at window ``L``.

    ``r_univariate`` is the max over channels of each channel's own effective rank;
    ``r_multivariate`` is the effective rank of the joint block-Hankel matrix. Under
    ``shared`` dependence the two are close; under ``independent`` the multivariate one
    is several times larger, which is the structural fact H1 has to reckon with.
    """
    p = signal.shape[1]
    cap = max(2, int(RANK_CAP_FRACTION * L))
    r_uni = max(
        oracle_rank(trajectory_matrix(signal[:, j], L), var_share, cap=cap)
        for j in range(p)
    )
    H_multi, _ = mssa_trajectory_matrix([signal[:, j] for j in range(p)], L)
    r_multi = oracle_rank(H_multi, var_share, cap=cap)
    return int(r_uni), int(r_multi)


def _true_left_subspace(clean_signal: np.ndarray, L: int, r: int, mode: str, channel: int = 0):
    """Leading-r left subspace of the CLEAN trajectory matrix.

    May return fewer than ``r`` columns: ``StandardSVD`` drops numerically-zero singular
    triples, so an exactly low-rank clean signal caps the truth at its algebraic rank.
    Callers must not paper over that — see :func:`_compare_subspace`.
    """
    if mode == "multivariate":
        H, _ = mssa_trajectory_matrix(
            [clean_signal[:, j] for j in range(clean_signal.shape[1])], L
        )
    else:
        H = trajectory_matrix(clean_signal[:, channel], L)
    return StandardSVD(rank=r).decompose(H).U


def _compare_subspace(U_est: np.ndarray, U_true: np.ndarray, r: int) -> float:
    """Subspace error between an estimate and the truth, at a comparable dimension.

    ``subspace_distance`` uses ``min(dim A, dim B)`` principal angles, so comparing an
    r-dimensional estimate against a lower-dimensional truth measures only *containment*
    and returns 0 whenever the truth sits inside the estimate — it silently flatters the
    estimate rather than erroring. That happens for real: with one r shared across
    heterogeneous channels (r=4 from a trended factor, applied to pure-sinusoid channels
    of rank 2) the univariate truth is 2-dimensional.

    Both sides are therefore clipped to the dimension the truth actually has. Recovering
    dimensions the clean signal does not possess is not a meaningful thing to score.
    """
    d = min(r, U_true.shape[1], U_est.shape[1])
    return subspace_distance(U_est[:, :d], U_true[:, :d])


def evaluate(
    X: np.ndarray,
    clean_signal: np.ndarray,
    backend_factory,
    L: int,
    r: int,
    mode: str,
) -> tuple[float, float]:
    """Fit once, return ``(signal_recovery_error, subspace_error)`` for one cell.

    Both metrics score against the known clean signal. The subspace error compares
    like with like: each estimate is measured against the true subspace *of its own
    mode and rank*, so a multivariate fit at r=10 is not penalised for not matching a
    univariate r=4 subspace.
    """
    p = X.shape[1]

    if mode == "multivariate":
        model = MSSA(window=L, backend=backend_factory(r)).fit(X)
        rec = np.asarray(model.reconstruct_full()).T  # (p, T) -> (T, p)
        U_true = _true_left_subspace(clean_signal, L, r, "multivariate")
        return (
            signal_recovery_error(rec, clean_signal),
            _compare_subspace(model.decomposition.U, U_true, r),
        )

    cols, subs = [], []
    for j in range(p):
        model = MSSA(window=L, backend=backend_factory(r)).fit(X[:, j])
        cols.append(model.reconstruct_full())
        U_true = _true_left_subspace(clean_signal, L, r, "univariate", channel=j)
        subs.append(_compare_subspace(model.decomposition.U, U_true, r))
    return signal_recovery_error(np.column_stack(cols), clean_signal), float(np.mean(subs))


def fmt_table(rows: list[dict], columns: list[str], widths: dict | None = None) -> str:
    """Minimal markdown table writer (no pandas dependency in the experiment layer)."""
    widths = widths or {}
    head = "| " + " | ".join(c.ljust(widths.get(c, len(c))) for c in columns) + " |"
    rule = "|" + "|".join("-" * (widths.get(c, len(c)) + 2) for c in columns) + "|"
    body = [
        "| " + " | ".join(str(row.get(c, "")).ljust(widths.get(c, len(c))) for c in columns) + " |"
        for row in rows
    ]
    return "\n".join([head, rule, *body])
