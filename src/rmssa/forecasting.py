"""Stage 5 — recurrent (M)SSA forecasting.

The Phase-2 study scored *model fit* only. The supervisor's 25 Jul 2026 note asked
whether the comparison is "in terms of model fit or forecasting"; this module adds the
forecasting arm so both are scored inside one design.

The recurrent forecast (Golyandina, Nekrutkin & Zhigljavsky 2001, §2.1) turns the
leading left singular subspace into a linear recurrence relation and rolls the
*reconstructed* series forward with it. Writing ``U`` for the (L x r) leading left
vectors, ``pi`` for their last row and ``U_`` for the first ``L-1`` rows:

    nu^2 = ||pi||^2,        R = U_ @ pi / (1 - nu^2)          (length L-1)
    g_{N+1} = R . (g_{N-L+2}, ..., g_N)

so one step is a dot product against the last ``L-1`` reconstructed values, iterated
for longer horizons. The recursion is *verticality*-conditioned: it needs nu^2 < 1.

Robust forecasting comes for free. Nothing here inspects which backend produced ``U``,
so a robust decomposition yields a robust LRR applied to a robust (outlier-cleaned)
reconstruction — the Rodrigues et al. (2020) robust forecasting recipe — while the
classical backend yields the textbook forecast. That is the point of keeping the
backend behind :class:`~rmssa.decomposition.DecompositionBackend`.

Univariate vs multivariate is the second axis. In horizontal MSSA every channel shares
one row space, so a *single* LRR derived from the common ``U`` drives every series
(Golyandina & Stepanov 2005); the univariate case fits and forecasts each series with
its own LRR. That contrast is exactly the "univariate vs multivariate" arm of the 2x2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .decomposition import DecompositionBackend
from .mssa import MSSA

__all__ = [
    "recurrent_coefficients",
    "verticality",
    "lrr_roots",
    "max_root_modulus",
    "is_explosive",
    "forecast_recurrent",
    "forecast",
    "rolling_origin_forecast",
    "RollingForecastResult",
]

#: An LRR may legitimately grow — AirPassengers trends upward, so its characteristic
#: polynomial has a root just above 1. What is not legitimate is growth by orders of
#: magnitude across the forecast horizon. A forecast is called explosive when the
#: dominant root implies growth beyond this factor over h steps.
DEFAULT_GROWTH_CAP = 10.0


def verticality(U: np.ndarray) -> float:
    """``nu^2``, the squared norm of the last row of ``U``.

    The recurrent forecast exists iff ``nu^2 < 1``. Values close to 1 mean the
    subspace is nearly "vertical" and the LRR is ill-conditioned.
    """
    U = np.asarray(U, dtype=float)
    pi = U[-1, :]
    return float(pi @ pi)


def recurrent_coefficients(U: np.ndarray, *, max_verticality: float = 1 - 1e-8) -> np.ndarray:
    """LRR coefficients ``R`` (length L-1) from the leading left subspace ``U`` (L x r).

    ``R`` is ordered so that the next value is ``R @ g[-(L-1):]`` — i.e. aligned with
    the last ``L-1`` values of the series in chronological order.
    """
    U = np.asarray(U, dtype=float)
    if U.ndim != 2:
        raise ValueError(f"U must be 2-D (L, r), got shape {U.shape}")
    if U.shape[0] < 2:
        raise ValueError("need window L >= 2 to build a recurrence")

    pi = U[-1, :]
    nu2 = float(pi @ pi)
    if nu2 >= max_verticality:
        raise ValueError(
            f"verticality nu^2 = {nu2:.12f} >= 1: the recurrent forecast is not defined "
            "for this subspace (try a different rank r or window L)."
        )
    return (U[:-1, :] @ pi) / (1.0 - nu2)


def lrr_roots(R: np.ndarray) -> np.ndarray:
    """Roots of the LRR's characteristic polynomial.

    With ``y_n = sum_m R[m] y_{n-(L-1)+m}`` the characteristic equation is
    ``z^{L-1} - sum_m R[m] z^m = 0``. The root moduli govern the forecast's long-run
    behaviour: modulus 1 is a pure oscillation or constant, above 1 is growth, below 1
    is decay.
    """
    R = np.asarray(R, dtype=float).ravel()
    return np.roots(np.concatenate([[1.0], -R[::-1]]))


def max_root_modulus(R: np.ndarray) -> float:
    """Largest characteristic-root modulus — the per-step growth factor."""
    roots = lrr_roots(R)
    return float(np.abs(roots).max()) if roots.size else 0.0


def is_explosive(R: np.ndarray, h: int, growth_cap: float = DEFAULT_GROWTH_CAP) -> bool:
    """Would this LRR grow by more than ``growth_cap`` over ``h`` steps?

    The failure mode this catches is real and specific: a near-vertical subspace makes
    the ``1/(1 - nu^2)`` factor in :func:`recurrent_coefficients` enormous. On a short
    training window a robust fit can land at ``nu^2 = 0.997``, giving a dominant root of
    1.34 and a forecast 12 steps later that is ~140x the series' own scale. Left
    unguarded, one such origin turns a rolling-origin RMSE into 1e8 and hides every
    other cell in the table.
    """
    if h <= 0:
        return False
    growth = max_root_modulus(R) ** h
    return bool(not np.isfinite(growth) or growth > growth_cap)


def forecast_recurrent(series: np.ndarray, R: np.ndarray, h: int) -> np.ndarray:
    """Continue ``series`` ``h`` steps with the LRR coefficients ``R``.

    ``series`` should be the *reconstructed* (signal) series, not the raw noisy one:
    the LRR governs the signal subspace.
    """
    g = np.asarray(series, dtype=float).ravel()
    R = np.asarray(R, dtype=float).ravel()
    m = R.size
    if g.size < m:
        raise ValueError(f"series has {g.size} points, need at least L-1 = {m}")
    if h < 0:
        raise ValueError(f"h must be >= 0, got {h}")

    out = np.empty(h)
    buf = g[-m:].copy()  # rolling window of the last L-1 values
    for i in range(h):
        nxt = float(R @ buf)
        out[i] = nxt
        buf = np.roll(buf, -1)
        buf[-1] = nxt
    return out


def forecast(
    model: MSSA,
    h: int,
    *,
    rank: int | None = None,
    growth_cap: float | None = None,
) -> np.ndarray:
    """``h``-step recurrent forecast from a fitted :class:`~rmssa.mssa.MSSA`.

    Returns ``(h,)`` for a univariate fit and ``(p, h)`` for an MSSA fit, where the
    common LRR is applied to each channel's own reconstruction.

    ``rank`` optionally truncates the subspace used to build the LRR (defaults to every
    component the backend kept). Forecasting usually wants the *signal* rank, so pass
    the same ``r`` used for the robust fit.

    ``growth_cap`` enables the stability guard: if the LRR would grow by more than this
    factor over ``h`` steps the forecast is refused with a ``ValueError`` rather than
    returned as a number many orders of magnitude off. ``None`` (default) disables it,
    keeping the plain textbook behaviour.
    """
    decomposition = model.decomposition
    U = decomposition.U
    if rank is not None:
        U = U[:, : min(rank, U.shape[1])]

    R = recurrent_coefficients(U)
    if growth_cap is not None and is_explosive(R, h, growth_cap):
        raise ValueError(
            f"explosive recurrence: dominant root {max_root_modulus(R):.4f} implies "
            f"growth of {max_root_modulus(R) ** h:.3g}x over h={h} "
            f"(nu^2 = {verticality(U):.6f}); refusing to forecast."
        )
    recon = model.reconstruct_full()

    if not model.multivariate_:
        return forecast_recurrent(np.asarray(recon), R, h)

    recon = np.asarray(recon)  # (p, N)
    return np.vstack([forecast_recurrent(recon[j], R, h) for j in range(recon.shape[0])])


# --------------------------------------------------------------- rolling evaluation
@dataclass
class RollingForecastResult:
    """Rolling-origin forecast errors for one (method, mode) configuration.

    ``errors`` is (n_origins, h_max, p): signed forecast error against the *clean*
    signal, so RMSE/MAE at any horizon is a slice away. NaN marks an origin where the
    forecast could not be produced (an ill-conditioned or explosive LRR), which is
    counted in ``n_failed`` rather than silently dropped — the failure *rate* is itself
    a result, and averaging an exploded forecast into the RMSE would destroy the table.
    """

    errors: np.ndarray
    horizons: np.ndarray
    origins: np.ndarray
    n_failed: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def n_origins(self) -> int:
        return int(self.origins.size)

    @property
    def failure_rate(self) -> float:
        """Share of origins that produced no usable forecast."""
        return self.n_failed / self.n_origins if self.n_origins else 0.0

    def _slice(self, h: int | None) -> np.ndarray:
        return self.errors if h is None else self.errors[:, h - 1, :]

    def rmse(self, h: int | None = None) -> float:
        """RMSE pooled over origins/series, at horizon ``h`` (1-based) or over all.

        NaN when every origin was rejected — there is genuinely no error to report, and
        a cell that never produced a usable forecast should not masquerade as a zero.
        """
        e = self._slice(h)
        return float(np.sqrt(np.mean(e[~np.isnan(e)] ** 2))) if np.any(~np.isnan(e)) else float("nan")

    def mae(self, h: int | None = None) -> float:
        e = self._slice(h)
        return float(np.mean(np.abs(e[~np.isnan(e)]))) if np.any(~np.isnan(e)) else float("nan")

    def by_horizon(self) -> dict[int, tuple[float, float]]:
        """``{h: (rmse, mae)}`` for every horizon."""
        return {int(h): (self.rmse(int(h)), self.mae(int(h))) for h in self.horizons}


def rolling_origin_forecast(
    X: np.ndarray,
    clean_signal: np.ndarray,
    backend_factory,
    *,
    window: int,
    rank: int,
    mode: str = "multivariate",
    h_max: int = 12,
    n_origins: int = 5,
    min_train: int | None = None,
    growth_cap: float | None = DEFAULT_GROWTH_CAP,
) -> RollingForecastResult:
    """Rolling-origin forecast evaluation of one (method, mode) cell.

    At each origin ``t`` the model is refitted on the *contaminated* history
    ``X[:t]`` and forecasts ``h_max`` steps; the error is measured against the
    *clean* signal ``clean_signal[t:t+h_max]``. Scoring against the clean signal (not
    the contaminated observations) is what makes the comparison meaningful: a method
    should not be rewarded for predicting an outlier.

    Origins are spread evenly over the last part of the sample, leaving ``h_max``
    points after the final one.

    ``growth_cap`` guards against explosive recurrences (see :func:`is_explosive`);
    affected origins are counted in ``n_failed`` instead of contributing a meaningless
    error. Pass ``None`` to disable the guard and see the raw divergence.
    """
    X = np.asarray(X, dtype=float)
    clean_signal = np.asarray(clean_signal, dtype=float)
    if X.shape != clean_signal.shape:
        raise ValueError(f"X {X.shape} and clean_signal {clean_signal.shape} must match")
    T, p = X.shape

    if min_train is None:
        min_train = max(2 * window, int(0.6 * T))
    last_origin = T - h_max
    if last_origin <= min_train:
        raise ValueError(
            f"not enough data: T={T}, h_max={h_max}, min_train={min_train}. "
            "Reduce h_max/window or lengthen the series."
        )
    origins = np.unique(np.linspace(min_train, last_origin, n_origins).astype(int))

    errors = np.full((origins.size, h_max, p), np.nan)
    n_failed = 0

    for i, t in enumerate(origins):
        hist = X[:t]
        target = clean_signal[t : t + h_max]
        try:
            if mode == "multivariate":
                model = MSSA(window=window, backend=backend_factory(rank)).fit(hist)
                pred = forecast(model, h_max, rank=rank, growth_cap=growth_cap).T  # (h, p)
            else:
                cols = []
                for j in range(p):
                    m = MSSA(window=window, backend=backend_factory(rank)).fit(hist[:, j])
                    cols.append(forecast(m, h_max, rank=rank, growth_cap=growth_cap))
                pred = np.column_stack(cols)  # (h, p)
        except (ValueError, np.linalg.LinAlgError):
            n_failed += 1
            continue
        errors[i] = pred - target

    return RollingForecastResult(
        errors=errors,
        horizons=np.arange(1, h_max + 1),
        origins=origins,
        n_failed=n_failed,
        meta={"mode": mode, "window": window, "rank": rank, "min_train": int(min_train),
              "growth_cap": growth_cap},
    )


def make_backend_factory(backend_cls: type[DecompositionBackend], **kwargs):
    """``rank -> backend`` factory, for passing a configured backend into the rollers."""
    return lambda r: backend_cls(rank=r, **kwargs)
