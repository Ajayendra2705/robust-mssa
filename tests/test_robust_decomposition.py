"""Phase-2 tests (Days 13-15): the robust SVD backends RobRSVD and AlternatingL1SVD.

Two properties the supervisor's directive hinges on:
  1. On CLEAN data the robust backends reproduce the standard SVD (weights -> 1),
     so "classical vs robust" is a fair comparison that collapses at epsilon = 0.
  2. Under a planted outlier the robust leading subspace stays closer to the
     clean-signal subspace than the standard SVD's, i.e. the outlier is
     down-weighted rather than allowed to rotate the factors.
Both plug into MSSA behind the unchanged decompose() contract (modularity).
"""

import numpy as np
import pytest

from rmssa.embedding import trajectory_matrix, mssa_trajectory_matrix
from rmssa.decomposition import (
    StandardSVD,
    RobustSVD,
    RobRSVD,
    AlternatingL1SVD,
    DecompositionBackend,
    Decomposition,
)
from rmssa.mssa import MSSA

ROBUST_BACKENDS = [RobRSVD, AlternatingL1SVD]


def subspace_distance(A: np.ndarray, B: np.ndarray) -> float:
    """sin of the largest principal angle between the column spans of A and B.

    0 == identical subspace, 1 == orthogonal. Robust to sign/rotation within the
    subspace, which is what we need when comparing leading factor spaces.
    """
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    cos_min = np.clip(s.min(), 0.0, 1.0)
    return float(np.sqrt(max(0.0, 1.0 - cos_min**2)))


# --------------------------------------------------------------------------- API
@pytest.mark.parametrize("Backend", ROBUST_BACKENDS)
def test_robust_backend_is_subclass_and_callable(Backend):
    b = Backend()
    assert isinstance(b, DecompositionBackend)
    assert isinstance(b, RobustSVD)
    f = np.sin(np.linspace(0, 6 * np.pi, 60))
    H = trajectory_matrix(f, 20)
    d1, d2 = b(H), b.decompose(H)
    assert np.allclose(d1.reconstruct_matrix(), d2.reconstruct_matrix())


@pytest.mark.parametrize("Backend", ROBUST_BACKENDS)
def test_returns_valid_decomposition(Backend):
    rng = np.random.default_rng(0)
    H = trajectory_matrix(rng.standard_normal(60), 20)
    d = Backend(rank=4).decompose(H)
    assert isinstance(d, Decomposition)
    assert d.rank == 4
    assert np.all(d.s > 0)
    assert np.all(np.diff(d.s) <= 1e-9)  # descending
    # unit-norm factor vectors (robust deflation keeps u, v unit-norm)
    assert np.allclose(np.linalg.norm(d.U, axis=0), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(d.Vt, axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("Backend", ROBUST_BACKENDS)
def test_invalid_rank_raises(Backend):
    with pytest.raises(ValueError):
        Backend(rank=0)


@pytest.mark.parametrize("Backend", ROBUST_BACKENDS)
def test_iteration_diagnostics(Backend):
    # n_iter_ / converged_ are exposed after decompose. On an exactly low-rank clean
    # matrix the fit converges quickly and well within max_iter.
    rng = np.random.default_rng(9)
    H = (rng.standard_normal((30, 4)) @ rng.standard_normal((4, 90)))
    b = Backend(rank=4, max_iter=100, tol=1e-9)
    b.decompose(H)
    assert 1 <= b.n_iter_ <= 100
    assert b.converged_ is True
    # a hard iteration cap of 1 must not converge on contaminated data
    Hc = H.copy()
    Hc[0, 0] += 50.0
    capped = Backend(rank=4, max_iter=1, tol=1e-12)
    capped.decompose(Hc)
    assert capped.n_iter_ == 1


# ----------------------------------------------------- clean-data == standard SVD
@pytest.mark.parametrize("Backend", ROBUST_BACKENDS)
def test_matches_standard_svd_on_exactly_low_rank(Backend):
    # On an EXACTLY rank-r matrix the rank-r residual is zero -> every robust weight
    # is exactly 1 -> the IRLS-ALS optimum IS the L2 SVD. Tight agreement expected.
    rng = np.random.default_rng(7)
    L, K, r = 30, 120, 4
    A = rng.standard_normal((L, r))
    B = rng.standard_normal((r, K))
    H = A @ B  # exact rank 4

    std = StandardSVD(rank=r).decompose(H)
    rob = Backend(rank=r).decompose(H)

    assert np.allclose(rob.s, std.s, rtol=1e-6, atol=1e-8)
    rel = np.linalg.norm(rob.reconstruct_matrix() - std.reconstruct_matrix()) / np.linalg.norm(H)
    assert rel < 1e-8
    assert subspace_distance(rob.U, std.U) < 1e-6


@pytest.mark.parametrize("Backend", ROBUST_BACKENDS)
def test_near_standard_svd_on_realistic_clean_signal(Backend):
    # On realistic clean data (a sampled sinusoid has a long tail of tiny singular
    # values) robust != L2 exactly, but they stay close: the property is that the
    # classical-vs-robust gap is negligible at contamination epsilon = 0.
    f = np.sin(np.linspace(0, 8 * np.pi, 160)) + 0.4 * np.cos(np.linspace(0, 3 * np.pi, 160))
    H = trajectory_matrix(f, 40)
    r = 4
    std = StandardSVD(rank=r).decompose(H)
    rob = Backend(rank=r).decompose(H)

    rel = np.linalg.norm(rob.reconstruct_matrix() - std.reconstruct_matrix()) / np.linalg.norm(H)
    assert rel < 1e-6
    # dominant, well-separated leading pair agrees up to sign/rotation
    assert subspace_distance(rob.U[:, :2], std.U[:, :2]) < 1e-5


@pytest.mark.parametrize("Backend", ROBUST_BACKENDS)
def test_full_rank_reconstruction_identity_clean(Backend):
    # With enough layers the robust fit reconstructs a clean low-rank H exactly.
    f = np.sin(np.linspace(0, 10 * np.pi, 120))  # SSA-rank 2
    H = trajectory_matrix(f, 30)
    d = Backend(rank=2).decompose(H)
    assert np.allclose(d.reconstruct_matrix(), H, atol=1e-6)


# ----------------------------------------------------- robustness to an outlier
@pytest.mark.parametrize("Backend", ROBUST_BACKENDS)
def test_downweights_planted_outlier(Backend):
    # Clean low-rank signal, then corrupt a single observation. Fitting rank-r on
    # the corrupted matrix, the robust reconstruction should recover the CLEAN
    # trajectory better than the standard SVD (which chases the outlier).
    t = np.linspace(0, 12 * np.pi, 200)
    clean = np.sin(t) + 0.5 * np.sin(0.5 * t)
    L, r = 40, 4
    H_clean = trajectory_matrix(clean, L)

    corrupted = clean.copy()
    corrupted[100] += 20.0 * clean.std()  # one gross outlier
    H = trajectory_matrix(corrupted, L)

    recon_std = StandardSVD(rank=r).decompose(H).reconstruct_matrix()
    recon_rob = Backend(rank=r).decompose(H).reconstruct_matrix()

    err_std = np.linalg.norm(recon_std - H_clean)
    err_rob = np.linalg.norm(recon_rob - H_clean)
    # robust reconstruction is closer to the clean signal
    assert err_rob < err_std


# ------------------------------------------------------------- MSSA integration
@pytest.mark.parametrize("Backend", ROBUST_BACKENDS)
def test_plugs_into_mssa_unchanged(Backend):
    # Modularity claim: swapping the backend needs no change to MSSA.
    rng = np.random.default_rng(1)
    series = [np.sin(np.linspace(0, 6 * np.pi, 120)) + 0.1 * rng.standard_normal(120)
              for _ in range(3)]
    model = MSSA(window=30, backend=Backend(rank=5)).fit(series)
    comps = model.reconstruct({"trend": [0, 1]})
    assert comps["trend"].shape == (3, 120)


@pytest.mark.parametrize("Backend", ROBUST_BACKENDS)
def test_mssa_trajectory_clean_agreement(Backend):
    # On a clean multivariate panel the robust MSSA factors match classical MSSA.
    series = [np.sin(np.linspace(0, 5 * np.pi, 90) + phase) for phase in (0.0, 0.7, 1.4)]
    H, _ = mssa_trajectory_matrix(series, 24)
    r = 3
    std = StandardSVD(rank=r).decompose(H)
    rob = Backend(rank=r).decompose(H)
    rel = np.linalg.norm(rob.reconstruct_matrix() - std.reconstruct_matrix()) / np.linalg.norm(H)
    assert rel < 1e-6


# ------------------------------------------------------- two algorithms agree clean
def test_both_robust_algorithms_agree_on_clean_data():
    # RobRSVD (Huber) and AlternatingL1SVD (L1) differ only in the weight function;
    # on clean data both reduce to ~the standard SVD, so they closely agree.
    f = np.cos(np.linspace(0, 7 * np.pi, 140))
    H = trajectory_matrix(f, 35)
    a = RobRSVD(rank=2).decompose(H)
    b = AlternatingL1SVD(rank=2).decompose(H)
    assert np.allclose(a.s, b.s, rtol=1e-4, atol=1e-6)
    rel = np.linalg.norm(a.reconstruct_matrix() - b.reconstruct_matrix()) / np.linalg.norm(H)
    assert rel < 1e-6


# ------------------------------------------------------------- initialisation basin
# The IRLS-by-imputation iteration is a fixed-point scheme, not a descent method, so
# the starting point decides which fixed point it reaches. Starting at the classical
# SVD of a heavily contaminated matrix starts it inside the basin the outliers have
# already rotated it into, and imputation cannot leave it (down-weighted cells are
# replaced by the current model's own values). The default init runs both the classical
# and a Winsorized start and keeps whichever has the lower objective at a common scale.


def _contaminated_narrow_fixture(seed: int):
    """A narrow, heavily contaminated trajectory matrix + its clean counterpart.

    Narrow because that is where the classical start fails most often: with few
    trajectory columns the outliers carry enough of the leading subspace to capture it.
    """
    from rmssa.datasets import make_synthetic_panel

    sp = make_synthetic_panel(T=60, p=2, k=2, noise_sd=0.03, contamination=0.10,
                              outlier_scale=15.0, seed=seed)
    H, _ = mssa_trajectory_matrix([sp.X[:, j] for j in range(2)], 40)
    Hc, _ = mssa_trajectory_matrix([sp.signal[:, j] for j in range(2)], 40)
    return H, Hc


def test_default_init_escapes_the_outlier_basin():
    # seed 0 is one of the seeds where the classical start is captured: it lands
    # ~orthogonal to the true factor subspace and stays there.
    H, Hc = _contaminated_narrow_fixture(seed=0)
    truth = np.linalg.svd(Hc, full_matrices=False)[0][:, :2]

    trapped = RobRSVD(rank=2, max_iter=300, init="classical").decompose(H)
    backend = RobRSVD(rank=2, max_iter=300)
    default = backend.decompose(H)

    assert subspace_distance(trapped.U, truth) > 0.5     # captured by the outliers
    assert subspace_distance(default.U, truth) < 0.15    # recovers the true subspace
    assert backend.init_used_ == "robust"


@pytest.mark.parametrize("Backend", ROBUST_BACKENDS)
def test_default_init_keeps_exactness_on_clean_low_rank_data(Backend):
    # The Winsorized start alone would break this: clipping perturbs an exactly
    # low-rank matrix off its exact solution, and the iteration sticks there. Scoring
    # both starts on a common objective must pick the classical one, whose residual
    # (and objective) is exactly zero.
    rng = np.random.default_rng(7)
    H = rng.standard_normal((30, 4)) @ rng.standard_normal((4, 120))  # exact rank 4

    std = StandardSVD(rank=4).decompose(H)
    backend = Backend(rank=4)
    rob = backend.decompose(H)

    assert backend.init_used_ == "classical"
    assert np.allclose(rob.s, std.s, rtol=1e-6, atol=1e-8)


def test_init_argument_is_validated():
    with pytest.raises(ValueError, match="init must be"):
        RobRSVD(rank=2, init="winsorised")
