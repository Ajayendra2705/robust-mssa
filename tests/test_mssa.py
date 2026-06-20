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


def test_mssa_accepts_2d_array_T_by_p():
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((70, 4))  # (T, p): 70 time points, 4 series
    model = MSSA(window=20).fit(arr)
    assert model.n_channels_ == 4
    # reconstruct_full returns (p, T); compare against arr.T
    assert np.allclose(model.reconstruct_full(), arr.T, atol=1e-9)


def test_mssa_accepts_dataframe():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.standard_normal((80, 3)), columns=["a", "b", "c"])
    model = MSSA(window=25).fit(df)
    assert model.n_channels_ == 3  # columns are channels
    assert np.allclose(model.reconstruct_full(), df.to_numpy().T, atol=1e-9)


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


def test_group_wcorrelation_univariate():
    n = 300
    t = np.arange(n)
    f = 0.02 * t + np.sin(2 * np.pi * t / 30)
    model = MSSA(window=60).fit(f)
    W = model.group_wcorrelation({"trend": [0], "season": [1, 2]})
    assert W.shape == (2, 2)
    assert np.allclose(np.diag(W), 1.0)
    assert abs(W[0, 1]) < 0.2  # trend vs season weakly correlated


def test_group_wcorrelation_mssa_averages_channels():
    rng = np.random.default_rng(4)
    t = np.arange(200)
    shared = np.sin(2 * np.pi * t / 25)
    channels = [0.01 * t + shared + 0.05 * rng.standard_normal(200) for _ in range(3)]
    model = MSSA(window=50).fit(channels)
    W = model.group_wcorrelation({"trend": [0], "rest": list(range(1, 6))})
    assert W.shape == (2, 2)
    assert np.allclose(np.diag(W), 1.0)
    assert np.allclose(W, W.T)


def test_reconstruct_before_fit_raises():
    with pytest.raises(RuntimeError):
        MSSA(window=10).reconstruct_full()
