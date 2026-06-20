"""Validation tests (Day 9): analytic SSA-rank ground truth + external cross-check.

These pin the correctness of the decomposition against results that do not depend on our
own code:

1. Analytic SSA ranks. A series that is a sum of products of polynomials, exponentials
   and sinusoids has a known, finite SSA-rank (Golyandina et al. 2001):
       - exponential a^t                      -> rank 1
       - sinusoid sin(wt)                      -> rank 2
       - linear trend a*t + b                  -> rank 2
       - sum of two distinct exponentials      -> rank 2
       - exp-modulated cosine a^t cos(wt)      -> rank 2
   We check the number of singular values above a relative threshold.

2. Cross-implementation check against ``pyts`` (skipped if not installed): the elementary
   reconstructed components and the full reconstruction must agree.
"""

import numpy as np
import pytest

from rmssa.embedding import trajectory_matrix
from rmssa.decomposition import StandardSVD
from rmssa.grouping import elementary_series


def _numerical_rank(f, L, rel=1e-9):
    s = np.linalg.svd(trajectory_matrix(f, L), compute_uv=False)
    return int(np.sum(s > rel * s[0]))


@pytest.mark.parametrize(
    "builder, expected_rank",
    [
        (lambda t: 0.95 ** t, 1),                                  # exponential
        (lambda t: np.sin(2 * np.pi * t / 17), 2),                 # sinusoid
        (lambda t: 0.5 + 0.03 * t, 2),                             # linear trend
        (lambda t: 0.9 ** t + 1.05 ** t, 2),                       # two exponentials
        (lambda t: 0.97 ** t * np.cos(2 * np.pi * t / 23), 2),     # modulated cosine
    ],
)
def test_analytic_ssa_rank(builder, expected_rank):
    t = np.arange(200)
    f = builder(t).astype(float)
    assert _numerical_rank(f, L=50) == expected_rank


def test_constant_series_rank_one():
    f = np.full(120, 3.0)
    assert _numerical_rank(f, L=40) == 1


def test_sum_of_two_sinusoids_rank_four():
    t = np.arange(400)
    f = np.sin(2 * np.pi * t / 13) + np.sin(2 * np.pi * t / 41)
    assert _numerical_rank(f, L=120) == 4


# --------------------------------------------------------------- external cross-check
# Gated behind the ``external`` marker (deselected by default) because importing pyts
# and its numba JIT warmup are slow. Run with:  pytest -m external
# pyts is imported *inside* each test so the default fast run never loads it.


@pytest.mark.external
def test_pyts_full_reconstruction_agrees():
    pytest.importorskip("pyts", reason="pyts not installed")
    from pyts.decomposition import SingularSpectrumAnalysis

    rng = np.random.default_rng(0)
    t = np.arange(120)
    f = 0.02 * t + np.sin(2 * np.pi * t / 20) + 0.1 * rng.standard_normal(120)
    L = 30

    # pyts: one group per elementary component -> sum reconstructs the series
    ssa = SingularSpectrumAnalysis(window_size=L, groups=L)
    pyts_components = ssa.fit_transform(f.reshape(1, -1))[0]  # (n_splits, N)
    assert np.allclose(pyts_components.sum(axis=0), f, atol=1e-8)

    # our full reconstruction also returns the series
    d = StandardSVD().decompose(trajectory_matrix(f, L))
    ours_full = elementary_series(d).sum(axis=0)
    assert np.allclose(ours_full, f, atol=1e-8)


@pytest.mark.external
def test_pyts_leading_component_matches_ours():
    pytest.importorskip("pyts", reason="pyts not installed")
    from pyts.decomposition import SingularSpectrumAnalysis

    t = np.arange(150)
    f = 0.01 * t + np.sin(2 * np.pi * t / 25)  # trend + one sinusoid
    L = 40

    ssa = SingularSpectrumAnalysis(window_size=L, groups=L)
    pyts_components = ssa.fit_transform(f.reshape(1, -1))[0]  # (n_splits, N)

    d = StandardSVD().decompose(trajectory_matrix(f, L))
    ours = elementary_series(d, n_components=pyts_components.shape[0])

    # Leading component (the trend) should be (near) identical up to sign.
    corr = abs(np.corrcoef(ours[0], pyts_components[0])[0, 1])
    assert corr > 0.999
