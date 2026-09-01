"""Outlier models for the simulation study (design v2).

The Phase-2 study used a single contamination type — isolated additive outliers at
random cells. The supervisor's 25 Jul 2026 note asked for "several types of outliers
and different percentages of outliers", so this module implements the four standard
time-series outlier types (Fox 1972; Chen & Liu 1993) behind one interface:

  * ``additive``     — isolated spikes at random cells (the Phase-2 case).
  * ``patch``        — runs of ``patch_len`` consecutive contaminated points, i.e. an
                       outlier *episode* rather than a single tick. Harder for robust
                       estimators: a patch looks locally like signal.
  * ``level_shift``  — a permanent step from a random time onward. The classic
                       structural break (the Nile series has a real one at 1899).
  * ``innovational`` — a shock entering the dynamics and decaying geometrically
                       (``phi ** k``), so it perturbs a whole neighbourhood.

All four fill the *same* cell budget ``eps * T * p``, so the contamination rate means
the same thing across types and the types are directly comparable at equal eps. The
realised rate is reported back (patches/shifts can overlap) rather than assumed.

**One caveat on eps for level shifts.** A permanent step starting in the middle of a
series alters every later point in its column, so with p = 4 a *single* shift already
touches ~15% of the panel's cells. Target rates below that are simply not expressible:
eps = 1% and eps = 5% both realise 15%. Where the number of structural breaks is the
quantity of interest, pass ``n_events`` instead of relying on ``rate`` — it sets the
event count directly for the three episodic models.

Magnitudes are expressed in units of each column's own standard deviation, so a panel
with heterogeneous column scales (real base series) is contaminated comparably.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["CONTAMINATION_KINDS", "Contamination", "contaminate"]

#: the four outlier types, in the order they are reported in the design table
CONTAMINATION_KINDS = ("additive", "patch", "level_shift", "innovational")


@dataclass
class Contamination:
    """Result of contaminating a clean panel.

    Attributes
    ----------
    X        : (T, p) contaminated panel (clean input + additive perturbation).
    outliers : (T, p) the perturbation itself.
    mask     : (T, p) boolean, True where a cell was altered.
    kind     : which outlier model was used.
    realised_rate : fraction of cells actually altered (may differ slightly from the
        requested eps for the episodic types, where events can overlap).
    meta     : type-specific parameters actually used (patch length, decay, ...).
    """

    X: np.ndarray
    outliers: np.ndarray
    mask: np.ndarray
    kind: str
    realised_rate: float
    meta: dict = field(default_factory=dict)


def _column_scales(signal: np.ndarray) -> np.ndarray:
    """Per-column standard deviation, with a safe fallback for constant columns."""
    sd = np.asarray(signal, dtype=float).std(axis=0)
    fallback = float(np.asarray(signal).std()) or 1.0
    sd = np.where(sd > 0, sd, fallback)
    return sd


# --------------------------------------------------------------------------- models
def _additive(out, mask, sd, budget, rng, **_):
    T, p = out.shape
    n = min(budget, T * p)
    flat = rng.choice(T * p, size=n, replace=False)
    rows, cols = np.unravel_index(flat, (T, p))
    signs = rng.choice([-1.0, 1.0], size=n)
    out[rows, cols] += signs * sd[cols]
    mask[rows, cols] = True
    return {}


def _patch(out, mask, sd, budget, rng, patch_len: int = 5, n_events: int | None = None, **_):
    T, p = out.shape
    patch_len = int(min(patch_len, T))
    n_patches = int(n_events) if n_events is not None else max(1, int(round(budget / patch_len)))
    for _ in range(n_patches):
        col = int(rng.integers(p))
        t0 = int(rng.integers(0, T - patch_len + 1))
        sign = float(rng.choice([-1.0, 1.0]))
        out[t0 : t0 + patch_len, col] += sign * sd[col]
        mask[t0 : t0 + patch_len, col] = True
    return {"patch_len": patch_len, "n_patches": n_patches}


def _level_shift(out, mask, sd, budget, rng, shift_window=(0.15, 0.85),
                 n_events: int | None = None, **_):
    """Permanent steps. Break points are drawn away from the very ends, otherwise a
    shift at t~0 or t~T is indistinguishable from a mean change / a single spike.

    ``n_events`` places exactly that many breaks and ignores the cell budget — the
    honest control for this type, since one step already saturates small eps.
    """
    T, p = out.shape
    lo = max(1, int(shift_window[0] * T))
    hi = max(lo + 1, int(shift_window[1] * T))
    filled, n_shifts = 0, 0
    limit = int(n_events) if n_events is not None else 10 * p + 10
    for _ in range(limit):
        if n_events is None and filled >= budget:
            break
        col = int(rng.integers(p))
        t0 = int(rng.integers(lo, hi))
        sign = float(rng.choice([-1.0, 1.0]))
        out[t0:, col] += sign * sd[col]
        filled += int((~mask[t0:, col]).sum())
        mask[t0:, col] = True
        n_shifts += 1
    return {"n_shifts": n_shifts}


def _innovational(out, mask, sd, budget, rng, phi: float = 0.7, floor: float = 0.05,
                  n_events: int | None = None, **_):
    """Shocks entering the dynamics, decaying as ``phi ** k`` until below ``floor``."""
    T, p = out.shape
    tail = max(1, int(np.ceil(np.log(floor) / np.log(phi))))
    decay = phi ** np.arange(tail)
    n_shocks = int(n_events) if n_events is not None else max(1, int(round(budget / tail)))
    for _ in range(n_shocks):
        col = int(rng.integers(p))
        t0 = int(rng.integers(0, T))
        sign = float(rng.choice([-1.0, 1.0]))
        m = min(tail, T - t0)
        out[t0 : t0 + m, col] += sign * sd[col] * decay[:m]
        mask[t0 : t0 + m, col] = True
    return {"phi": phi, "tail": tail, "n_shocks": n_shocks}


_MODELS = {
    "additive": _additive,
    "patch": _patch,
    "level_shift": _level_shift,
    "innovational": _innovational,
}


# ------------------------------------------------------------------------ interface
def contaminate(
    signal: np.ndarray,
    *,
    kind: str = "additive",
    rate: float = 0.0,
    magnitude: float = 8.0,
    rng: np.random.Generator | int | None = None,
    **kwargs,
) -> Contamination:
    """Contaminate a clean (T, p) panel with one of :data:`CONTAMINATION_KINDS`.

    Parameters
    ----------
    signal : (T, p) clean panel to perturb (also sets the per-column scale).
    kind : outlier model; see module docstring.
    rate : eps, the target fraction of altered cells.
    magnitude : outlier size in units of the column's own standard deviation.
    rng : Generator or seed.
    **kwargs : forwarded to the model (``patch_len``, ``phi``, ``shift_window``, and
        ``n_events`` to set the event count directly instead of deriving it from
        ``rate`` — the only workable control for ``level_shift``, see module docstring).

    Returns
    -------
    Contamination with ``X = signal + outliers``, the mask, and the realised rate.
    """
    if kind not in _MODELS:
        raise ValueError(f"unknown contamination kind {kind!r}; expected one of {CONTAMINATION_KINDS}")
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")

    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2-D (T, p), got shape {signal.shape}")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    T, p = signal.shape
    out = np.zeros((T, p))
    mask = np.zeros((T, p), dtype=bool)
    meta: dict = {}

    budget = int(round(rate * T * p))
    # n_events drives the episodic models directly, so it must fire even at rate 0
    if budget > 0 or kwargs.get("n_events"):
        # magnitude scales the per-column sd once, so the models only see "sd[col]"
        scales = _column_scales(signal) * float(magnitude)
        meta = _MODELS[kind](out, mask, scales, budget, generator, **kwargs)

    return Contamination(
        X=signal + out,
        outliers=out,
        mask=mask,
        kind=kind,
        realised_rate=float(mask.mean()),
        meta={"magnitude": float(magnitude), "target_rate": float(rate), **meta},
    )
