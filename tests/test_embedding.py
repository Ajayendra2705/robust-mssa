"""Day-5 tests: embedding produces the correct (block-)Hankel trajectory matrix."""

import numpy as np
import pytest

from rmssa.embedding import trajectory_matrix, mssa_trajectory_matrix, n_columns, embed


def test_shape_is_L_by_K():
    f = np.arange(10.0)
    L = 4
    X = trajectory_matrix(f, L)
    assert X.shape == (L, n_columns(len(f), L))
    assert X.shape == (4, 7)


def test_hankel_entry_is_f_of_i_plus_j():
    f = np.arange(20.0) ** 1.3  # arbitrary distinct values
    L = 6
    X = trajectory_matrix(f, L)
    L_, K_ = X.shape
    for i in range(L_):
        for j in range(K_):
            assert X[i, j] == pytest.approx(f[i + j])


def test_hankel_constant_along_antidiagonals():
    f = np.random.default_rng(0).standard_normal(30)
    L = 8
    X = trajectory_matrix(f, L)
    L_, K_ = X.shape
    for s in range(L_ + K_ - 1):
        vals = [X[i, s - i] for i in range(L_) if 0 <= s - i < K_]
        assert np.allclose(vals, vals[0])


def test_first_column_and_first_row():
    f = np.arange(12.0)
    L = 5
    X = trajectory_matrix(f, L)
    assert np.allclose(X[:, 0], f[:L])          # first column = first window
    assert np.allclose(X[0, :], f[: X.shape[1]])  # first row = leading values


def test_embed_alias_matches():
    f = np.linspace(-1, 1, 15)
    assert np.allclose(embed(f, 5), trajectory_matrix(f, 5))


def test_output_is_independent_copy():
    f = np.arange(10.0)
    X = trajectory_matrix(f, 4)
    X[0, 0] = 999.0
    assert f[0] == 0.0  # mutating the trajectory matrix must not touch the series


@pytest.mark.parametrize("L", [1, 0, 10, 11, -2])
def test_invalid_window_raises(L):
    f = np.arange(10.0)
    with pytest.raises((ValueError, TypeError)):
        trajectory_matrix(f, L)


def test_mssa_block_structure():
    rng = np.random.default_rng(1)
    series = [rng.standard_normal(20), rng.standard_normal(20), rng.standard_normal(20)]
    L = 6
    H, widths = mssa_trajectory_matrix(series, L)
    K = n_columns(20, L)
    p = len(series)
    assert H.shape == (L, p * K)
    assert widths == [K, K, K]
    # each width-K block equals the corresponding channel's trajectory matrix
    for j, s in enumerate(series):
        block = H[:, j * K : (j + 1) * K]
        assert np.allclose(block, trajectory_matrix(s, L))


def test_mssa_unequal_lengths():
    rng = np.random.default_rng(2)
    series = [rng.standard_normal(20), rng.standard_normal(25)]
    L = 6
    H, widths = mssa_trajectory_matrix(series, L)
    assert widths == [n_columns(20, L), n_columns(25, L)]
    assert H.shape == (L, sum(widths))


def test_mssa_empty_raises():
    with pytest.raises(ValueError):
        mssa_trajectory_matrix([], 4)
