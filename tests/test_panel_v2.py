"""Tests for the design-v2 panel generator: real base series + dependence factor."""

import numpy as np
import pytest

from rmssa.datasets import (
    BASE_SERIES,
    DEPENDENCE_LEVELS,
    factor_bank,
    load_base_series,
    make_panel,
    make_synthetic_panel,
)


@pytest.mark.parametrize("name", [n for n in BASE_SERIES if n != "synthetic"])
def test_base_series_load_offline(name):
    s = load_base_series(name)
    assert s.ndim == 1 and s.size >= 100
    assert np.isfinite(s).all()


def test_airpassengers_is_the_canonical_series():
    s = load_base_series("airpassengers")
    assert s.size == 144
    assert s[0] == 112.0 and s[-1] == 432.0


def test_unknown_base_series_rejected():
    with pytest.raises(ValueError, match="unknown base series"):
        load_base_series("not_a_series")


# ------------------------------------------------------------------- dependence
def test_shared_dependence_gives_rank_k():
    pan = make_panel(T=200, p=6, k=2, base="synthetic", dependence="shared", seed=0)
    assert pan.signal_rank == 2


def test_independent_dependence_gives_full_rank():
    pan = make_panel(T=200, p=6, k=2, base="synthetic", dependence="independent", seed=0)
    assert pan.signal_rank == 6


@pytest.mark.parametrize("base,T", [("synthetic", 300), ("airpassengers", 144),
                                    ("sunspots", 300), ("co2", 600)])
def test_independent_has_no_systematic_comovement(base, T):
    """Independence is a property of the *generating process*, and the statistic that
    tests it is the SIGNED correlation, which must average to zero.

    The absolute correlation does NOT go to zero at finite T for an autocorrelated base
    (mean |corr| ~ 0.35 for airpassengers at T=144) — Yule's (1926) nonsense-correlation
    effect, since low-frequency power shrinks the effective sample size. That gap between
    "independent by construction" and "uncorrelated in sample" is why both statistics are
    carried on Panel and reported in the results.
    """
    corrs = []
    for seed in range(40):
        pan = make_panel(T=T, p=4, k=2, base=base, dependence="independent", seed=seed)
        C = np.corrcoef(pan.signal, rowvar=False)
        corrs.extend(C[np.triu_indices(4, 1)])
    assert abs(float(np.mean(corrs))) < 0.08


def test_shared_is_more_correlated_than_independent():
    ind = make_panel(T=300, p=6, k=2, base="synthetic", dependence="independent", seed=0)
    shr = make_panel(T=300, p=6, k=2, base="synthetic", dependence="shared", seed=0)
    assert shr.mean_abs_corr > ind.mean_abs_corr


def test_shared_ratio_dials_dependence_monotonically():
    lo = make_panel(T=300, p=6, k=2, dependence="partial", shared_ratio=0.1, seed=0)
    hi = make_panel(T=300, p=6, k=2, dependence="partial", shared_ratio=0.9, seed=0)
    assert hi.mean_abs_corr > lo.mean_abs_corr


@pytest.mark.parametrize("dependence", DEPENDENCE_LEVELS)
def test_signal_columns_are_standardised(dependence):
    """Unit-sd columns are what make eps and magnitude mean the same thing everywhere."""
    pan = make_panel(T=200, p=5, k=2, dependence=dependence, seed=1)
    assert np.allclose(pan.signal.std(axis=0), 1.0, atol=1e-8)


def test_ground_truth_decomposition_is_consistent():
    pan = make_panel(T=200, p=5, k=2, contamination=0.05, seed=2)
    assert np.allclose(pan.X, pan.signal + pan.noise + pan.outliers)
    assert pan.X.shape == pan.signal.shape == (200, 5)


@pytest.mark.parametrize("dependence", DEPENDENCE_LEVELS)
@pytest.mark.parametrize("base", ["synthetic", "sunspots"])
def test_signal_equals_factors_times_loadings(dependence, base):
    """The documented ground-truth invariant. It is easy to break: the generator
    rescales signal columns to unit sd, and a rescale absorbed into the signal but not
    the loadings leaves `loadings` quietly wrong while every experiment still passes,
    because the experiments read `signal` and never `loadings`."""
    pan = make_panel(T=200, p=4, k=2, base=base, dependence=dependence, seed=0)
    m = pan.loadings.shape[1]
    assert np.allclose(pan.factors[:, :m] @ pan.loadings.T, pan.signal, atol=1e-10)


@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9])
def test_shared_ratio_is_exactly_the_shared_variance_share(rho):
    """`partial` dependence must mean what it says: rho is the fraction of each series'
    variance carried by the common factors."""
    pan = make_panel(T=400, p=4, k=2, base="synthetic", dependence="partial",
                     shared_ratio=rho, seed=0)
    k = 2
    shared = pan.factors[:, :k] @ pan.loadings[:, :k].T
    idio = pan.factors[:, k : k + 4] @ pan.loadings[:, k:].T
    share = (shared.var(axis=0) / (shared.var(axis=0) + idio.var(axis=0))).mean()
    assert share == pytest.approx(rho, abs=1e-6)


def test_airpassengers_matches_the_published_series():
    """Guards the hard-coded copy against a typo: Box & Jenkins totals."""
    s = load_base_series("airpassengers")
    assert s.sum() == 40363.0
    assert s.mean() == pytest.approx(280.2986, abs=1e-4)
    # annual totals rise every year in the real series
    assert np.all(np.diff(s.reshape(12, 12).sum(axis=1)) > 0)


@pytest.mark.parametrize("base,T", [("synthetic", 300), ("airpassengers", 144),
                                    ("nile", 100), ("sunspots", 300), ("co2", 600)])
def test_every_base_builds_a_panel(base, T):
    pan = make_panel(T=T, p=4, k=2, base=base, contamination=0.05, seed=0)
    assert pan.X.shape == (T, 4)
    assert np.isfinite(pan.X).all()


def test_base_shorter_than_T_is_reported_clearly():
    """The real bank caps T: nile is 100 points, so asking for T=144 must say so."""
    with pytest.raises(ValueError, match="only 100 points"):
        make_panel(T=144, p=3, base="nile")


def test_bad_dependence_and_ratio_rejected():
    with pytest.raises(ValueError, match="dependence must be"):
        make_panel(dependence="sometimes")
    with pytest.raises(ValueError, match="shared_ratio must be"):
        make_panel(dependence="partial", shared_ratio=2.0)


# ----------------------------------------------------------------- factor bank
def test_surrogates_preserve_the_amplitude_spectrum():
    """Same spectrum => same autocovariance => same SSA-rank profile as the original."""
    F, _ = factor_bank(144, 4, base="airpassengers", method="surrogate", rng=0)
    spectra = [np.abs(np.fft.rfft(F[:, j] - F[:, j].mean())) for j in range(4)]
    for j in range(1, 4):
        # surrogates are standardised copies, so compare shape via correlation
        assert np.corrcoef(spectra[0], spectra[j])[0, 1] > 0.99


def test_segment_method_returns_distinct_real_windows():
    F, labels = factor_bank(144, 4, base="airpassengers", method="segment")
    assert len(set(labels)) == 4
    assert F.shape == (144, 4)


def test_segment_method_reports_when_bank_too_small():
    with pytest.raises(ValueError, match="cannot supply"):
        factor_bank(2000, 12, base="airpassengers", method="segment")


def test_bad_bank_method_rejected():
    with pytest.raises(ValueError, match="method must be"):
        factor_bank(100, 2, base="nile", method="teleport")


# ------------------------------------------------------------- back-compatibility
def test_phase2_generator_still_reproduces_bitwise():
    """make_synthetic_panel is frozen: the Phase-2 results already sent to the
    supervisor must keep reproducing exactly."""
    a = make_synthetic_panel(T=100, p=4, k=2, contamination=0.05, seed=0)
    b = make_synthetic_panel(T=100, p=4, k=2, contamination=0.05, seed=0)
    assert np.array_equal(a.X, b.X)
    assert a.mask.sum() == 20
