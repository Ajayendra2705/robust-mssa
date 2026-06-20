"""Day-8 tests: synthetic panel generator (offline; no network)."""

import numpy as np

from rmssa.datasets import make_synthetic_panel


def test_shapes_and_ground_truth():
    sp = make_synthetic_panel(T=200, p=5, k=2, seed=0)
    assert sp.X.shape == (200, 5)
    assert sp.factors.shape == (200, 2)
    assert sp.loadings.shape == (5, 2)
    # clean signal is exactly rank k
    assert np.linalg.matrix_rank(sp.signal, tol=1e-8) == 2


def test_clean_panel_has_no_outliers():
    sp = make_synthetic_panel(contamination=0.0, seed=1)
    assert sp.mask.sum() == 0
    assert np.allclose(sp.outliers, 0.0)
    assert sp.contamination_rate == 0.0


def test_contamination_rate_matches_request():
    sp = make_synthetic_panel(T=100, p=10, contamination=0.1, seed=2)
    assert sp.mask.sum() == 100  # 0.1 * 100 * 10
    assert abs(sp.contamination_rate - 0.1) < 1e-9
    # outliers only where mask is set
    assert np.all(sp.outliers[~sp.mask] == 0.0)


def test_reproducible_with_seed():
    a = make_synthetic_panel(seed=7).X
    b = make_synthetic_panel(seed=7).X
    assert np.allclose(a, b)
