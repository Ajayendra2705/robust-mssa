"""Day-5 tests: the StandardSVD backend and the Decomposition container."""

import numpy as np
import pytest

from rmssa.embedding import trajectory_matrix, mssa_trajectory_matrix
from rmssa.decomposition import StandardSVD, Decomposition, DecompositionBackend


def test_backend_is_subclass_and_callable():
    backend = StandardSVD()
    assert isinstance(backend, DecompositionBackend)
    f = np.sin(np.linspace(0, 6 * np.pi, 60))
    H = trajectory_matrix(f, 20)
    # __call__ and decompose agree
    d1, d2 = backend(H), backend.decompose(H)
    assert np.allclose(d1.reconstruct_matrix(), d2.reconstruct_matrix())


def test_decomposition_reconstructs_matrix_full_rank():
    f = np.sin(np.linspace(0, 6 * np.pi, 80)) + 0.3 * np.cos(np.linspace(0, 2 * np.pi, 80))
    H = trajectory_matrix(f, 24)
    d = StandardSVD().decompose(H)
    assert np.allclose(d.reconstruct_matrix(), H, atol=1e-9)


def test_sum_of_elementary_matrices_equals_H():
    rng = np.random.default_rng(0)
    H = trajectory_matrix(rng.standard_normal(50), 15)
    d = StandardSVD().decompose(H)
    acc = sum(d.elementary(i) for i in range(d.rank))
    assert np.allclose(acc, H, atol=1e-9)


def test_singular_values_sorted_and_positive():
    rng = np.random.default_rng(1)
    H = trajectory_matrix(rng.standard_normal(40), 12)
    d = StandardSVD().decompose(H)
    assert np.all(d.s > 0)
    assert np.all(np.diff(d.s) <= 1e-12)  # descending


def test_orthonormality():
    rng = np.random.default_rng(2)
    H = trajectory_matrix(rng.standard_normal(60), 20)
    d = StandardSVD().decompose(H)
    assert np.allclose(d.U.T @ d.U, np.eye(d.rank), atol=1e-9)
    assert np.allclose(d.Vt @ d.Vt.T, np.eye(d.rank), atol=1e-9)


def test_contributions_sum_to_one():
    f = np.sin(np.linspace(0, 8 * np.pi, 100))
    H = trajectory_matrix(f, 30)
    d = StandardSVD().decompose(H)
    c = d.contributions()
    assert c.sum() == pytest.approx(1.0)
    assert np.all(c >= 0)


def test_pure_sinusoid_is_low_rank():
    # A single sinusoid has SSA-rank 2; leading two components dominate.
    f = np.sin(np.linspace(0, 10 * np.pi, 200))
    H = trajectory_matrix(f, 50)
    d = StandardSVD().decompose(H)
    assert d.contributions()[:2].sum() > 0.999


def test_rank_truncation():
    rng = np.random.default_rng(3)
    H = trajectory_matrix(rng.standard_normal(80), 25)
    d = StandardSVD(rank=3).decompose(H)
    assert d.rank == 3
    assert d.U.shape[1] == 3 and d.Vt.shape[0] == 3 and d.s.shape[0] == 3
    # rank-r reconstruction is the best rank-r L2 approximation -> not exact
    assert not np.allclose(d.reconstruct_matrix(), H)


def test_low_rank_truncation_matches_eckart_young():
    rng = np.random.default_rng(4)
    H = trajectory_matrix(rng.standard_normal(70), 20)
    r = 5
    d = StandardSVD(rank=r).decompose(H)
    # Best rank-r error equals the (r+1)-th singular value (Eckart-Young).
    full = np.linalg.svd(H, compute_uv=False)
    err = np.linalg.norm(H - d.reconstruct_matrix(), 2)
    assert err == pytest.approx(full[r], rel=1e-6)


def test_invalid_rank_raises():
    with pytest.raises(ValueError):
        StandardSVD(rank=0)


def test_decomposition_shape_property():
    H = trajectory_matrix(np.arange(40.0), 12)
    d = StandardSVD().decompose(H)
    assert d.shape == H.shape


def test_decomposition_validates_consistency():
    with pytest.raises(ValueError):
        Decomposition(U=np.zeros((4, 3)), s=np.ones(2), Vt=np.zeros((3, 5)))


def test_mssa_decomposition_roundtrip():
    rng = np.random.default_rng(5)
    series = [rng.standard_normal(40) for _ in range(4)]
    H, _ = mssa_trajectory_matrix(series, 12)
    d = StandardSVD().decompose(H)
    assert d.shape == H.shape
    assert np.allclose(d.reconstruct_matrix(), H, atol=1e-9)
