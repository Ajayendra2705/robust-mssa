"""Tests for the four outlier models of design v2."""

import numpy as np
import pytest

from rmssa.contamination import CONTAMINATION_KINDS, contaminate


def clean_panel(T=200, p=5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    return np.column_stack([np.sin(2 * np.pi * t / (20 + 5 * j)) + rng.normal(0, 0.01, T)
                            for j in range(p)])


@pytest.mark.parametrize("kind", CONTAMINATION_KINDS)
def test_zero_rate_leaves_panel_untouched(kind):
    S = clean_panel()
    c = contaminate(S, kind=kind, rate=0.0, rng=0)
    assert c.mask.sum() == 0
    assert np.allclose(c.X, S)
    assert c.realised_rate == 0.0


@pytest.mark.parametrize("kind", CONTAMINATION_KINDS)
def test_outliers_only_where_masked(kind):
    S = clean_panel()
    c = contaminate(S, kind=kind, rate=0.05, magnitude=8.0, rng=1)
    assert np.all(c.outliers[~c.mask] == 0.0)
    assert np.allclose(c.X - S, c.outliers)


@pytest.mark.parametrize("kind", CONTAMINATION_KINDS)
def test_realised_rate_is_near_target(kind):
    """All four models fill the same cell budget, so eps is comparable across types."""
    S = clean_panel(T=300, p=8)
    c = contaminate(S, kind=kind, rate=0.05, rng=2)
    # episodic types (patches/shifts) can overlap, so allow a modest band
    assert 0.03 <= c.realised_rate <= 0.09


def test_additive_hits_the_budget_exactly():
    S = clean_panel(T=100, p=10)
    c = contaminate(S, kind="additive", rate=0.1, rng=3)
    assert c.mask.sum() == 100


def test_magnitude_scales_with_column_sd():
    """A column with 10x the scale gets 10x the outlier, so eps means one thing."""
    T = 200
    t = np.arange(T)
    S = np.column_stack([np.sin(t / 5.0), 10.0 * np.sin(t / 5.0)])
    c = contaminate(S, kind="additive", rate=0.5, magnitude=4.0, rng=4)
    peak = [np.abs(c.outliers[c.mask[:, j], j]).max() for j in range(2)]
    assert peak[1] == pytest.approx(10 * peak[0], rel=1e-9)


def test_patch_makes_runs_not_isolated_points():
    S = clean_panel(T=400, p=4)
    c = contaminate(S, kind="patch", rate=0.05, rng=5, patch_len=8)
    # every contaminated column should contain at least one run of >= 2
    runs = []
    for j in range(4):
        col = c.mask[:, j]
        if col.any():
            longest = max(len(r) for r in "".join("1" if v else "0" for v in col).split("0") if r)
            runs.append(longest)
    assert runs and max(runs) >= 4


def test_level_shift_persists_to_the_end():
    S = clean_panel(T=300, p=4)
    c = contaminate(S, kind="level_shift", rate=0.05, rng=6)
    # a shifted column stays shifted: the last row is always contaminated in any
    # column that is contaminated at all
    for j in range(4):
        if c.mask[:, j].any():
            assert c.mask[-1, j]


def test_innovational_decays_away_from_the_shock():
    S = np.zeros((200, 1))
    S[:, 0] = np.sin(np.arange(200) / 7.0)
    c = contaminate(S, kind="innovational", rate=0.05, magnitude=8.0, rng=7, phi=0.7)
    mags = np.abs(c.outliers[c.mask[:, 0], 0])
    # a geometric tail means many distinct magnitudes below the peak, not a flat block
    assert mags.size > 1
    assert mags.min() < 0.5 * mags.max()


def test_rejects_unknown_kind_and_bad_rate():
    S = clean_panel()
    with pytest.raises(ValueError, match="unknown contamination kind"):
        contaminate(S, kind="nonsense", rate=0.1)
    with pytest.raises(ValueError, match="rate must be"):
        contaminate(S, kind="additive", rate=1.5)


def test_level_shift_eps_saturates_which_is_why_n_events_exists():
    """A permanent step covers the rest of its column, so with few series a single
    shift already exceeds a small target eps — 1% and 5% realise the same rate."""
    S = clean_panel(T=144, p=4)
    one = contaminate(S, kind="level_shift", rate=0.01, rng=8)
    five = contaminate(S, kind="level_shift", rate=0.05, rng=8)
    assert one.realised_rate == five.realised_rate > 0.10


@pytest.mark.parametrize("kind,key", [("patch", "n_patches"),
                                      ("level_shift", "n_shifts"),
                                      ("innovational", "n_shocks")])
def test_n_events_sets_the_event_count_directly(kind, key):
    S = clean_panel(T=300, p=4)
    c = contaminate(S, kind=kind, rate=0.0, magnitude=8.0, rng=9, n_events=3)
    assert c.meta[key] == 3
    assert c.mask.any()  # fires even though rate == 0


def test_more_level_shifts_contaminate_more():
    S = clean_panel(T=300, p=6)
    few = contaminate(S, kind="level_shift", rate=0.0, rng=10, n_events=1)
    many = contaminate(S, kind="level_shift", rate=0.0, rng=10, n_events=4)
    assert many.realised_rate > few.realised_rate


def test_reproducible_under_the_same_seed():
    S = clean_panel()
    a = contaminate(S, kind="patch", rate=0.05, rng=11)
    b = contaminate(S, kind="patch", rate=0.05, rng=11)
    assert np.array_equal(a.outliers, b.outliers)
