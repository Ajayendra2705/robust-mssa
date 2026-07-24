"""Day-12 tests: ground-truth & stability metrics."""

import numpy as np
import pytest

from rmssa import metrics


# ------------------------------------------------------------------- subspace
def test_identical_subspace_has_zero_distance():
    rng = np.random.default_rng(0)
    U = rng.standard_normal((20, 3))
    assert metrics.subspace_distance(U, U) == pytest.approx(0.0, abs=1e-6)
    assert metrics.subspace_overlap(U, U) == pytest.approx(1.0)
    assert metrics.grassmann_distance(U, U) == pytest.approx(0.0, abs=1e-6)


def test_rotation_within_subspace_is_invariant():
    # Rotating a basis within its own span must not change any subspace metric.
    rng = np.random.default_rng(1)
    U = rng.standard_normal((30, 4))
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))  # 4x4 rotation
    V = U @ Q
    assert metrics.subspace_distance(U, V) == pytest.approx(0.0, abs=1e-6)
    assert metrics.factor_stability(U, V) == pytest.approx(1.0)


def test_sign_and_permutation_invariance():
    rng = np.random.default_rng(2)
    U = rng.standard_normal((25, 3))
    V = U[:, [2, 0, 1]] * np.array([-1.0, 1.0, -1.0])  # permute + flip signs
    assert metrics.subspace_distance(U, V) == pytest.approx(0.0, abs=1e-6)


def test_orthogonal_subspaces_max_distance():
    L = 10
    A = np.eye(L)[:, :3]
    B = np.eye(L)[:, 3:6]  # orthogonal columns
    assert metrics.subspace_distance(A, B) == pytest.approx(1.0)
    assert metrics.subspace_overlap(A, B) == pytest.approx(0.0, abs=1e-12)


def test_principal_angles_known_value():
    # Two planes sharing one axis; the other axes at 60 degrees.
    A = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    c, s = np.cos(np.pi / 3), np.sin(np.pi / 3)
    B = np.array([[1.0, 0.0], [0.0, c], [0.0, s], [0.0, 0.0]])
    ang = metrics.principal_angles(A, B)
    assert np.allclose(np.sort(ang), [0.0, np.pi / 3], atol=1e-9)


def test_partial_overlap_between_zero_and_one():
    rng = np.random.default_rng(3)
    base = rng.standard_normal((15, 4))
    A = base[:, :2]
    B = np.column_stack([base[:, 1], rng.standard_normal(15)])  # shares one direction
    d = metrics.subspace_distance(A, B)
    assert 0.0 < d < 1.0


def test_accepts_1d_vector():
    v = np.arange(8.0)
    assert metrics.subspace_distance(v, 2 * v) == pytest.approx(0.0, abs=1e-6)


# -------------------------------------------------------------- reconstruction
def test_rmse_mae_zero_on_exact():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((10, 5))
    assert metrics.rmse(X, X) == 0.0
    assert metrics.mae(X, X) == 0.0
    assert metrics.relative_frobenius(X, X) == 0.0


def test_rmse_known_value():
    a = np.zeros((2, 2))
    b = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert metrics.rmse(a, b) == pytest.approx(1.0)
    assert metrics.mae(a, b) == pytest.approx(1.0)


def test_relative_frobenius_scale_free():
    rng = np.random.default_rng(5)
    tgt = rng.standard_normal((8, 8))
    est = tgt + 0.1 * rng.standard_normal((8, 8))
    r1 = metrics.relative_frobenius(est, tgt)
    r2 = metrics.relative_frobenius(10 * est, 10 * tgt)  # invariant to common scaling
    assert r1 == pytest.approx(r2, rel=1e-9)


def test_signal_recovery_error_is_relative_frobenius():
    rng = np.random.default_rng(6)
    S = rng.standard_normal((12, 4))
    est = S.copy()
    assert metrics.signal_recovery_error(est, S) == pytest.approx(0.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.rmse(np.zeros((2, 3)), np.zeros((2, 4)))


# ------------------------------------------------------------------ stability
def test_factor_stability_bounds():
    rng = np.random.default_rng(7)
    U = rng.standard_normal((20, 3))
    assert metrics.factor_stability(U, U) == pytest.approx(1.0)
    orth = np.eye(20)[:, 10:13]
    # a subspace orthogonal to U (span the last coords, U is generic dense -> not exactly 0,
    # so build U in the first 10 coords)
    U2 = np.zeros((20, 3))
    U2[:10, :] = rng.standard_normal((10, 3))
    assert metrics.factor_stability(U2, orth) == pytest.approx(0.0, abs=1e-12)
