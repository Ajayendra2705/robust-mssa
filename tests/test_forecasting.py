"""Tests for recurrent (M)SSA forecasting.

The core correctness property: a series that exactly satisfies a linear recurrence of
order < L (sinusoid, exponential, polynomial trend, and their sums/products) must be
continued *exactly* by the recurrent forecast. Everything else builds on that.
"""

import numpy as np
import pytest

from rmssa.datasets import make_panel
from rmssa.decomposition import AlternatingL1SVD, RobRSVD, StandardSVD
from rmssa.forecasting import (
    forecast,
    forecast_recurrent,
    is_explosive,
    lrr_roots,
    max_root_modulus,
    recurrent_coefficients,
    rolling_origin_forecast,
    verticality,
)
from rmssa.mssa import MSSA

T, L = 200, 50


def sinusoid(n=T, period=25.0, start=0):
    t = np.arange(start, start + n)
    return np.sin(2 * np.pi * t / period)


# ------------------------------------------------------------- exactness on LRRs
def test_forecast_continues_a_sinusoid_exactly():
    model = MSSA(window=L, backend=StandardSVD(rank=2)).fit(sinusoid())
    pred = forecast(model, 12, rank=2)
    assert np.allclose(pred, sinusoid(12, start=T), atol=1e-10)


def test_forecast_continues_trend_plus_modulated_sinusoid_exactly():
    t = np.arange(T)
    signal = np.exp(0.002 * t) * np.sin(2 * np.pi * t / 25) + 0.01 * t + 1.0
    model = MSSA(window=L, backend=StandardSVD(rank=6)).fit(signal)
    tt = np.arange(T, T + 12)
    truth = np.exp(0.002 * tt) * np.sin(2 * np.pi * tt / 25) + 0.01 * tt + 1.0
    assert np.allclose(forecast(model, 12, rank=6), truth, atol=1e-9)


def test_mssa_forecast_shape_and_exactness():
    t = np.arange(T)
    X = np.column_stack([np.sin(2 * np.pi * t / 25),
                         0.7 * np.cos(2 * np.pi * t / 25) + 0.3 * np.sin(2 * np.pi * t / 25)])
    model = MSSA(window=L, backend=StandardSVD(rank=2)).fit(X)
    pred = forecast(model, 6, rank=2)
    tt = np.arange(T, T + 6)
    truth = np.vstack([np.sin(2 * np.pi * tt / 25),
                       0.7 * np.cos(2 * np.pi * tt / 25) + 0.3 * np.sin(2 * np.pi * tt / 25)])
    assert pred.shape == (2, 6)
    assert np.allclose(pred, truth, atol=1e-10)


@pytest.mark.parametrize("backend", [RobRSVD, AlternatingL1SVD])
def test_robust_backends_forecast_clean_data_like_the_classical_one(backend):
    """No robustness tax on the forecast either: on clean data the robust LRR is the
    classical LRR, which is what makes the 2x2 comparison fair at eps = 0."""
    model = MSSA(window=L, backend=backend(rank=2, max_iter=60, tol=1e-9)).fit(sinusoid())
    assert np.allclose(forecast(model, 12, rank=2), sinusoid(12, start=T), atol=1e-8)


# ------------------------------------------------------------------- LRR mechanics
def test_coefficients_length_and_verticality():
    U = StandardSVD(rank=2).decompose(
        MSSA(window=L, backend=StandardSVD(rank=2)).fit(sinusoid()).H_
    ).U
    R = recurrent_coefficients(U)
    assert R.shape == (L - 1,)
    assert 0.0 <= verticality(U) < 1.0


def test_forecast_recurrent_is_iterated_one_step_ahead():
    g = sinusoid()
    model = MSSA(window=L, backend=StandardSVD(rank=2)).fit(g)
    R = recurrent_coefficients(model.decomposition.U[:, :2])
    step = forecast_recurrent(g, R, 1)
    three = forecast_recurrent(g, R, 3)
    assert np.allclose(step[0], three[0])
    # feeding the first prediction back must reproduce the rest
    assert np.allclose(forecast_recurrent(np.append(g, three[0]), R, 2), three[1:])


def test_zero_horizon_returns_empty():
    model = MSSA(window=L, backend=StandardSVD(rank=2)).fit(sinusoid())
    assert forecast(model, 0, rank=2).shape == (0,)


def test_vertical_subspace_is_reported_not_silently_wrong():
    U = np.zeros((5, 1))
    U[-1, 0] = 1.0  # nu^2 == 1 exactly
    with pytest.raises(ValueError, match="verticality"):
        recurrent_coefficients(U)


def test_series_shorter_than_the_recurrence_is_rejected():
    with pytest.raises(ValueError, match="at least"):
        forecast_recurrent(np.arange(3.0), np.ones(10), 1)


# ------------------------------------------------------------ stability guard
def test_sinusoid_lrr_has_unit_modulus_roots():
    """A pure oscillation neither grows nor decays: dominant root modulus == 1."""
    model = MSSA(window=L, backend=StandardSVD(rank=2)).fit(sinusoid())
    R = recurrent_coefficients(model.decomposition.U[:, :2])
    assert max_root_modulus(R) == pytest.approx(1.0, abs=1e-6)
    assert not is_explosive(R, 12)


def test_growing_signal_is_allowed_not_flagged():
    """A trending series legitimately has a root above 1 — the guard must not reject
    it, or AirPassengers-style data could never be forecast at all."""
    t = np.arange(T)
    model = MSSA(window=L, backend=StandardSVD(rank=4)).fit(np.exp(0.004 * t))
    R = recurrent_coefficients(model.decomposition.U[:, :4])
    assert max_root_modulus(R) > 1.0
    assert not is_explosive(R, 12)


def test_explosive_recurrence_is_detected():
    R = np.zeros(5)
    R[-1] = 3.0  # y_n = 3 y_{n-1}
    assert max_root_modulus(R) == pytest.approx(3.0)
    assert is_explosive(R, 12)
    assert not is_explosive(R, 0)


def test_lrr_roots_count_matches_order():
    R = np.array([0.1, -0.2, 0.3])
    assert lrr_roots(R).size == 3


def test_forecast_refuses_to_return_an_exploded_series():
    """The concrete failure this guards: a near-vertical robust subspace on a short
    window produced nu^2 = 0.997 and a forecast ~140x the series scale."""
    t = np.arange(120)
    noisy = np.sin(2 * np.pi * t / 25) + 0.4 * np.random.default_rng(0).standard_normal(120)
    model = MSSA(window=L, backend=StandardSVD(rank=40)).fit(noisy)
    R = recurrent_coefficients(model.decomposition.U[:, :40])
    if is_explosive(R, 12):  # high rank on noise usually is
        with pytest.raises(ValueError, match="explosive recurrence"):
            forecast(model, 12, rank=40, growth_cap=10.0)
        # without the guard the same call returns numbers, however useless
        assert np.isfinite(forecast(model, 12, rank=40)).all()


def test_guard_is_off_by_default_in_forecast():
    """forecast() keeps textbook behaviour unless a cap is asked for."""
    model = MSSA(window=L, backend=StandardSVD(rank=2)).fit(sinusoid())
    assert forecast(model, 5, rank=2).shape == (5,)


# --------------------------------------------------------------- rolling origin
def test_rolling_origin_shapes_and_clean_signal_target():
    pan = make_panel(T=300, p=4, k=2, base="synthetic", contamination=0.0, seed=0)
    res = rolling_origin_forecast(
        pan.X, pan.signal, lambda r: StandardSVD(rank=r),
        window=40, rank=6, mode="multivariate", h_max=6, n_origins=3,
    )
    assert res.errors.shape == (res.origins.size, 6, 4)
    assert res.n_failed == 0
    assert res.rmse() > 0 and res.mae() > 0
    assert set(res.by_horizon()) == {1, 2, 3, 4, 5, 6}


def test_rolling_origin_error_grows_with_horizon():
    pan = make_panel(T=300, p=3, k=2, base="synthetic", noise_sd=0.05, seed=1)
    res = rolling_origin_forecast(
        pan.X, pan.signal, lambda r: StandardSVD(rank=r),
        window=40, rank=6, mode="multivariate", h_max=12, n_origins=4,
    )
    assert res.rmse(12) > res.rmse(1)


def test_rolling_origin_rejects_impossible_split():
    pan = make_panel(T=60, p=2, k=1, base="synthetic", seed=0)
    with pytest.raises(ValueError, match="not enough data"):
        rolling_origin_forecast(pan.X, pan.signal, lambda r: StandardSVD(rank=r),
                                window=20, rank=4, h_max=40, n_origins=3)


def test_rolling_origin_rejects_mismatched_shapes():
    pan = make_panel(T=200, p=3, k=2, seed=0)
    with pytest.raises(ValueError, match="must match"):
        rolling_origin_forecast(pan.X, pan.signal[:, :2], lambda r: StandardSVD(rank=r),
                                window=30, rank=4, h_max=6, n_origins=2)


def test_mssa_shared_lrr_needs_the_JOINT_rank_for_independent_channels():
    """Horizontal MSSA drives every channel with ONE recurrence read off the common row
    space. That is valid for unrelated channels only if the retained rank covers their
    JOINT dimension: two independent rank-2 signals span 4, and at r < 4 the shared LRR
    cannot represent them.

    This is the noise-free version of the rank-capacity effect behind the H1 result, so
    it is pinned here where nothing else can explain it.
    """
    t = np.arange(300)
    a = np.sin(2 * np.pi * t / 25)
    b = np.sin(2 * np.pi * t / 13) + 0.5 * np.cos(2 * np.pi * t / 13)
    tt = np.arange(300, 312)
    truth = np.vstack([np.sin(2 * np.pi * tt / 25),
                       np.sin(2 * np.pi * tt / 13) + 0.5 * np.cos(2 * np.pi * tt / 13)])
    X = np.column_stack([a, b])

    starved = MSSA(window=60, backend=StandardSVD(rank=2)).fit(X)
    assert np.abs(forecast(starved, 12, rank=2) - truth).max() > 0.5

    adequate = MSSA(window=60, backend=StandardSVD(rank=4)).fit(X)
    assert np.allclose(forecast(adequate, 12, rank=4), truth, atol=1e-9)
