"""Day-6 tests: w-correlation weights and separability diagnostics."""

import numpy as np
import pytest

from rmssa.embedding import trajectory_matrix
from rmssa.decomposition import StandardSVD
from rmssa.grouping import (
    wcorr_weights,
    wcorrelation_matrix,
    elementary_wcorrelation,
    suggest_groups_by_contribution,
)


def test_wcorr_weights_shape_and_symmetry():
    n, L = 20, 8
    w = wcorr_weights(n, L)
    assert w.shape == (n,)
    assert np.array_equal(w, w[::-1])          # symmetric
    assert w[0] == 1 and w[-1] == 1            # corner anti-diagonals have length 1
    assert w.max() == min(L, n - L + 1)        # plateau at L* = min(L, K)


def test_wcorr_weights_sum_equals_LK():
    # total weight = number of matrix entries = L * K
    n, L = 30, 12
    assert wcorr_weights(n, L).sum() == L * (n - L + 1)


def test_wcorrelation_identity_diagonal():
    rng = np.random.default_rng(0)
    series = rng.standard_normal((4, 50))
    W = wcorrelation_matrix(series, window=15)
    assert np.allclose(np.diag(W), 1.0)
    assert np.allclose(W, W.T)
    assert np.all(np.abs(W) <= 1.0 + 1e-9)


def test_identical_series_wcorr_one():
    s = np.sin(np.linspace(0, 4 * np.pi, 60))
    W = wcorrelation_matrix(np.vstack([s, s.copy()]), window=20)
    assert W[0, 1] == pytest.approx(1.0)


def test_trend_and_oscillation_are_separable():
    # A trend component and a sinusoid component should have small |w-corr|.
    n = 200
    t = np.arange(n)
    f = 0.02 * t + np.sin(2 * np.pi * t / 40)
    d = StandardSVD().decompose(trajectory_matrix(f, 50))
    W = elementary_wcorrelation(d, window=50, n_components=6)
    # component 0 (trend) vs the sinusoid pair (1,2): weak correlation
    assert abs(W[0, 1]) < 0.2
    assert abs(W[0, 2]) < 0.2


def test_sinusoid_pair_is_correlated():
    # The two eigentriples of one sinusoid form a strongly-coupled pair.
    n = 300
    t = np.arange(n)
    f = np.sin(2 * np.pi * t / 30)
    d = StandardSVD().decompose(trajectory_matrix(f, 60))
    W = elementary_wcorrelation(d, window=60, n_components=4)
    assert abs(W[0, 1]) > 0.5  # leading pair strongly coupled


def test_suggest_groups_partitions_all_indices():
    rng = np.random.default_rng(3)
    d = StandardSVD().decompose(trajectory_matrix(rng.standard_normal(100), 30))
    g = suggest_groups_by_contribution(d, threshold=0.9)
    assert sorted(g["signal"] + g["noise"]) == list(range(d.rank))
