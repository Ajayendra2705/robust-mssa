"""Day-6 tests: diagonal averaging and the reconstruction identity."""

import numpy as np
import pytest

from rmssa.embedding import trajectory_matrix, mssa_trajectory_matrix
from rmssa.decomposition import StandardSVD
from rmssa.reconstruction import diagonal_average, reconstruct_series, reconstruct_mssa


def test_diagonal_average_inverts_embedding():
    f = np.sin(np.linspace(0, 6 * np.pi, 73)) + 0.1 * np.arange(73)
    X = trajectory_matrix(f, 20)
    assert np.allclose(diagonal_average(X), f, atol=1e-12)


def test_diagonal_average_simple_matrix():
    # [[1,2,3],[4,5,6]] anti-diagonals: {1},{2,4},{3,5},{6} -> 1,3,4,6
    Y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.allclose(diagonal_average(Y), [1.0, 3.0, 4.0, 6.0])


def test_reconstruction_identity_univariate():
    rng = np.random.default_rng(0)
    f = rng.standard_normal(120)
    d = StandardSVD().decompose(trajectory_matrix(f, 40))
    full = reconstruct_series(d)["all"]
    assert np.allclose(full, f, atol=1e-9)


def test_sum_of_groups_equals_original():
    f = np.sin(np.linspace(0, 8 * np.pi, 150)) + 0.2 * np.arange(150) / 150
    d = StandardSVD().decompose(trajectory_matrix(f, 50))
    groups = {"trend": [0], "rest": list(range(1, d.rank))}
    comps = reconstruct_series(d, groups)
    assert np.allclose(comps["trend"] + comps["rest"], f, atol=1e-9)


def test_reconstruction_identity_mssa():
    rng = np.random.default_rng(1)
    channels = [rng.standard_normal(80) for _ in range(3)]
    H, widths = mssa_trajectory_matrix(channels, 25)
    d = StandardSVD().decompose(H)
    rec = reconstruct_mssa(d, widths)["all"]  # (p, N)
    assert rec.shape == (3, 80)
    for j, c in enumerate(channels):
        assert np.allclose(rec[j], c, atol=1e-9)


def test_mssa_group_split_sums_to_original():
    rng = np.random.default_rng(2)
    channels = [rng.standard_normal(60) for _ in range(2)]
    H, widths = mssa_trajectory_matrix(channels, 20)
    d = StandardSVD().decompose(H)
    g = {"a": [0, 1], "b": list(range(2, d.rank))}
    rec = reconstruct_mssa(d, widths, g)
    total = rec["a"] + rec["b"]
    assert np.allclose(total, np.vstack(channels), atol=1e-9)


def test_diagonal_average_requires_2d():
    with pytest.raises(ValueError):
        diagonal_average(np.arange(5.0))
