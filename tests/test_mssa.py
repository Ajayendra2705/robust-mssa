"""Day-7 tests: the MSSA orchestrator end-to-end."""

import numpy as np
import pytest

from rmssa.mssa import MSSA
from rmssa.decomposition import StandardSVD


def test_univariate_full_reconstruction_identity():
    f = np.sin(np.linspace(0, 6 * np.pi, 120)) + 0.01 * np.arange(120)
    model = MSSA(window=40).fit(f)
    assert not model.multivariate_
    assert np.allclose(model.reconstruct_full(), f, atol=1e-9)


def test_mssa_full_reconstruction_identity():
    rng = np.random.default_rng(0)
    channels = [rng.standard_normal(90) for _ in range(3)]
    model = MSSA(window=30).fit(channels)
    assert model.multivariate_ and model.n_channels_ == 3
    rec = model.reconstruct_full()  # (p, N)
    assert rec.shape == (3, 90)
    assert np.allclose(rec, np.vstack(channels), atol=1e-9)


def test_mssa_accepts_2d_array():
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((4, 70))  # (p, N)
    model = MSSA(window=20).fit(arr)
    assert model.n_channels_ == 4
    assert np.allclose(model.reconstruct_full(), arr, atol=1e-9)


def test_grouped_reconstruction_sums_back():
    f = np.sin(np.linspace(0, 10 * np.pi, 200)) + 0.005 * np.arange(200)
    model = MSSA(window=50).fit(f)
    comps = model.reconstruct({"trend": [0], "rest": list(range(1, model.decomposition.rank))})
    assert np.allclose(comps["trend"] + comps["rest"], f, atol=1e-9)


def test_contributions_and_wcorr_shapes():
    rng = np.random.default_rng(0)
    t = np.arange(160)
    # full-rank signal (trend + sinusoid + noise) so >=5 components exist
    f = 0.01 * t + np.sin(2 * np.pi * t / 40) + 0.1 * rng.standard_normal(160)
    model = MSSA(window=40).fit(f)
    c = model.contributions()
    assert c.sum() == pytest.approx(1.0)
    W = model.wcorrelation(n_components=5)
    assert W.shape == (5, 5)
    assert np.allclose(np.diag(W), 1.0)


def test_from_config_and_custom_backend():
    # noisy signal so the numerical rank exceeds the requested truncation ranks
    rng = np.random.default_rng(1)
    f = np.sin(np.linspace(0, 6 * np.pi, 100)) + 0.1 * rng.standard_normal(100)
    m1 = MSSA.from_config({"window": 30, "rank": 4}).fit(f)
    assert m1.decomposition.rank == 4
    m2 = MSSA(window=30, backend=StandardSVD(rank=2)).fit(f)
    assert m2.decomposition.rank == 2


def test_reconstruct_before_fit_raises():
    with pytest.raises(RuntimeError):
        MSSA(window=10).reconstruct_full()
