"""Dataset loaders and synthetic panel generators.

Day-8 scope: a small real equity-index loader (Yahoo Finance, with on-disk caching and
a graceful offline fallback) and a synthetic panel generator with a *known* low-rank
factor structure and injectable outliers (used heavily in Phase-2 validation).

The synthetic generator returns ground truth (factors, loadings, clean signal, outlier
mask) so that subspace-recovery error can be computed exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .contamination import contaminate

__all__ = [
    "SyntheticPanel",
    "make_synthetic_panel",
    "load_yahoo",
    "DEFAULT_INDICES",
    # design v2
    "BASE_SERIES",
    "DEPENDENCE_LEVELS",
    "Panel",
    "load_base_series",
    "factor_bank",
    "make_panel",
]


# A small, diversified default panel for the Day-8 baseline demo.
DEFAULT_INDICES = {
    "^GSPC": "S&P 500",
    "^FTSE": "FTSE 100",
    "^N225": "Nikkei 225",
    "^GDAXI": "DAX",
}


# --------------------------------------------------------------------------- synthetic
@dataclass
class SyntheticPanel:
    """A synthetic panel X = S + N + O with known structure.

    Attributes
    ----------
    X        : (T, p) observed panel (signal + noise + outliers).
    signal   : (T, p) clean low-rank signal S = factors @ loadings.T.
    factors  : (T, k) latent factor series.
    loadings : (p, k) factor loadings.
    noise    : (T, p) idiosyncratic noise N.
    outliers : (T, p) sparse additive outliers O.
    mask     : (T, p) boolean mask of outlier locations.
    """

    X: np.ndarray
    signal: np.ndarray
    factors: np.ndarray
    loadings: np.ndarray
    noise: np.ndarray
    outliers: np.ndarray
    mask: np.ndarray

    @property
    def contamination_rate(self) -> float:
        return float(self.mask.mean())


def make_synthetic_panel(
    T: int = 400,
    p: int = 6,
    k: int = 2,
    noise_sd: float = 0.05,
    contamination: float = 0.0,
    outlier_scale: float = 8.0,
    periods: Sequence[float] | None = None,
    trend: bool = True,
    seed: int | None = 0,
) -> SyntheticPanel:
    """Generate a panel of p series driven by k shared latent factors.

    Factors are smooth (trend + sinusoids) so they have low SSA-rank; loadings are
    random. Outliers are placed at a ``contamination`` fraction of cells with magnitude
    ``outlier_scale`` x the signal's std and random sign.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    if periods is None:
        periods = [50.0, 25.0, 80.0][:k] or [50.0]

    # latent factors: optional shared linear trend + distinct sinusoids
    factors = np.empty((T, k))
    for f in range(k):
        comp = np.sin(2 * np.pi * t / periods[f % len(periods)])
        if trend and f == 0:
            comp = comp + 0.01 * t
        factors[:, f] = comp
    factors = (factors - factors.mean(0)) / (factors.std(0) + 1e-12)

    loadings = rng.standard_normal((p, k))
    signal = factors @ loadings.T  # (T, p)
    signal = signal / signal.std() * 1.0  # normalise overall scale

    noise = noise_sd * rng.standard_normal((T, p))

    outliers = np.zeros((T, p))
    mask = np.zeros((T, p), dtype=bool)
    if contamination > 0:
        n_out = int(round(contamination * T * p))
        flat = rng.choice(T * p, size=n_out, replace=False)
        rows, cols = np.unravel_index(flat, (T, p))
        signs = rng.choice([-1.0, 1.0], size=n_out)
        outliers[rows, cols] = signs * outlier_scale * signal.std()
        mask[rows, cols] = True

    X = signal + noise + outliers
    return SyntheticPanel(X, signal, factors, loadings, noise, outliers, mask)


# ============================================================================
# Design v2 (post 25 Jul 2026 supervisor note)
# ----------------------------------------------------------------------------
# Two things the Phase-2 generator above could not express, both requested by the
# supervisor:
#
#   1. **Real base series as the clean signal.** "Use simple real series, e.g.
#      AirPassengers, contaminated with several types of outliers." The original,
#      uncontaminated series plays the role of the known clean signal S, so
#      ground-truth scoring survives the move away from artificial data.
#   2. **Cross-series dependence as an explicit design factor.** The Phase-2
#      generator is always fully shared (every series loads on the same k factors),
#      which makes the supervisor's two stated expectations untestable:
#         H1  robust SSA ~= robust MSSA when the series are independent;
#         H2  all four combinations coincide with no contamination AND independence.
#      ``dependence`` in {independent, partial, shared} supplies the missing axis.
#
# ``make_synthetic_panel`` above is deliberately left untouched so the Phase-2
# numbers already reported to the supervisor still reproduce bit-for-bit.
# ============================================================================

#: real benchmark series usable as clean signals (all available offline)
BASE_SERIES = ("airpassengers", "nile", "sunspots", "co2", "synthetic")

#: cross-series dependence levels
DEPENDENCE_LEVELS = ("independent", "partial", "shared")

#: distinct sinusoid periods for the artificial factor bank (never aliasing)
_SYNTHETIC_PERIODS = (50.0, 30.0, 80.0, 20.0, 65.0, 40.0, 55.0, 35.0, 70.0, 25.0)

# Box & Jenkins (1976) airline passengers, monthly 1949-1960, 144 obs. Hard-coded
# because it is the supervisor's suggested primary series and statsmodels does not
# ship it; this keeps the whole simulation study runnable offline.
_AIRPASSENGERS = (
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
)


def load_base_series(name: str) -> np.ndarray:
    """A real benchmark series as a 1-D float array (no download required).

    The bank spans deliberately different dynamics, so a panel built from it is not
    secretly four copies of one process:

      * ``airpassengers`` (144, monthly) — strong trend + multiplicative seasonality;
      * ``nile``          (100, annual)  — near-level series with a real 1899 break;
      * ``sunspots``      (309, annual)  — quasi-periodic ~11-year cycle, no trend;
      * ``co2``           (2284, weekly) — dominant trend + clean annual seasonality.
    """
    key = name.lower()
    if key == "airpassengers":
        return np.asarray(_AIRPASSENGERS, dtype=float)
    if key == "nile":
        from statsmodels.datasets import nile

        return nile.load_pandas().data["volume"].to_numpy(dtype=float)
    if key == "sunspots":
        from statsmodels.datasets import sunspots

        return sunspots.load_pandas().data["SUNACTIVITY"].to_numpy(dtype=float)
    if key == "co2":
        from statsmodels.datasets import co2

        s = co2.load_pandas().data["co2"]
        return s.interpolate().bfill().to_numpy(dtype=float)
    raise ValueError(f"unknown base series {name!r}; expected one of {BASE_SERIES}")


def _standardise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-12)


def _synthetic_factor_bank(T: int, n_factors: int, trend: bool = True) -> tuple[np.ndarray, list[str]]:
    """Artificial factors: distinct-period sinusoids, factor 0 optionally trended."""
    t = np.arange(T)
    periods = list(_SYNTHETIC_PERIODS)
    while len(periods) < n_factors:  # extend the pool if a very wide panel is asked for
        periods.append(periods[-1] + 7.0)
    F = np.empty((T, n_factors))
    labels = []
    for j in range(n_factors):
        comp = np.sin(2 * np.pi * t / periods[j])
        if trend and j == 0:
            comp = comp + 0.01 * t
        F[:, j] = _standardise(comp)
        labels.append(f"sin{periods[j]:g}" + ("+trend" if trend and j == 0 else ""))
    return F, labels


def _phase_randomise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A phase-randomised surrogate of ``x`` (Theiler et al. 1992).

    Keeps the amplitude spectrum — hence the autocovariance — but replaces the Fourier
    phases with independent uniform draws, so two surrogates of one series are mutually
    independent by construction.

    **Caveat, measured not assumed:** matching the spectrum does *not* match the SSA
    rank profile. AirPassengers reaches 99% of its trajectory-matrix variance at r=10;
    a surrogate of it needs r=28. The original's low rank comes from *deterministic*
    trend-plus-seasonal structure, while a random-phase realisation with the same power
    spectrum spreads that energy over many more singular triples. Surrogates are
    therefore the right tool for the independence axis but the wrong one for anything
    that depends on the signal being genuinely low-rank — use ``method="segment"``
    (real series, real rank profile) there and accept the residual correlation.
    """
    n = x.size
    spec = np.fft.rfft(x)
    phases = rng.uniform(0.0, 2 * np.pi, spec.size)
    phases[0] = 0.0  # keep the mean real
    if n % 2 == 0:
        phases[-1] = 0.0  # Nyquist bin must stay real
    return np.fft.irfft(np.abs(spec) * np.exp(1j * phases), n=n)


def factor_bank(
    T: int,
    n_factors: int,
    base: str = "airpassengers",
    method: str = "surrogate",
    rng: np.random.Generator | int | None = None,
    names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """``n_factors`` mutually distinct standardised signals of length ``T``.

    ``base="synthetic"`` gives distinct-period sinusoids (never aliasing, so
    near-orthogonal). For a real base there are two ways to fill the bank:

    * ``method="surrogate"`` (default) — factor 0 is the real series itself; the rest
      are independent **phase-randomised surrogates** of it. Same spectral character,
      uncorrelated by construction, and available in any number.
    * ``method="segment"`` — non-overlapping length-``T`` windows taken round-robin
      across the real bank. Every value is genuinely observed data, but the factors are
      *not* independent: trending series co-trend (|corr| ~ 0.65 for airpassengers vs
      co2) and a strongly periodic series co-cycles with its own later segments
      (|corr| ~ 0.9 between co2 windows). Kept for the all-real robustness check;
      not suitable for the ``independent`` dependence level.

    Either way the realised dependence is measured, not assumed — see
    :attr:`Panel.max_abs_corr`.
    """
    if base == "synthetic":
        return _synthetic_factor_bank(T, n_factors)

    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    if method == "surrogate":
        series = load_base_series(base)
        if series.size < T:
            raise ValueError(
                f"base series {base!r} has only {series.size} points, need T={T}. "
                f"Use a longer base (co2, sunspots) or a smaller T."
            )
        head = series[:T]
        F = np.empty((T, n_factors))
        labels = []
        for j in range(n_factors):
            F[:, j] = _standardise(head if j == 0 else _phase_randomise(head, generator))
            labels.append(base if j == 0 else f"{base}~surr{j}")
        return F, labels

    if method != "segment":
        raise ValueError(f"method must be 'surrogate' or 'segment', got {method!r}")

    order = list(names) if names is not None else [base] + [n for n in BASE_SERIES if n not in (base, "synthetic")]
    loaded = {}
    for nm in order:
        series = load_base_series(nm)
        if series.size >= T:
            loaded[nm] = series

    # round-robin over segment index so the first factors come from *different* series
    segments: list[tuple[str, int]] = []
    for seg in range(max((loaded[nm].size // T) for nm in loaded) if loaded else 0):
        for nm in order:
            if nm in loaded and (seg + 1) * T <= loaded[nm].size:
                segments.append((nm, seg))

    if len(segments) < n_factors:
        available = {nm: loaded[nm].size // T for nm in loaded}
        raise ValueError(
            f"factor bank cannot supply {n_factors} distinct signals of length T={T}: "
            f"only {len(segments)} non-overlapping segments exist {available}. "
            f"Use a shorter T, a smaller p, method='surrogate', or base='synthetic'."
        )

    F = np.empty((T, n_factors))
    labels = []
    for j, (nm, seg) in enumerate(segments[:n_factors]):
        F[:, j] = _standardise(loaded[nm][seg * T : (seg + 1) * T])
        labels.append(f"{nm}[{seg}]")
    return F, labels


@dataclass
class Panel:
    """A design-v2 panel ``X = S + N + O`` with known clean signal and known structure.

    Same ground-truth contract as :class:`SyntheticPanel` (so the Phase-2 metrics apply
    unchanged), plus the design coordinates that produced it.
    """

    X: np.ndarray
    signal: np.ndarray
    factors: np.ndarray
    loadings: np.ndarray
    noise: np.ndarray
    outliers: np.ndarray
    mask: np.ndarray
    base: str = "synthetic"
    dependence: str = "shared"
    kind: str = "additive"
    factor_labels: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def contamination_rate(self) -> float:
        return float(self.mask.mean())

    def _abs_corrs(self) -> np.ndarray:
        """|correlation| over the distinct pairs of *clean* series."""
        if self.signal.shape[1] < 2:
            return np.zeros(1)
        C = np.abs(np.corrcoef(self.signal, rowvar=False))
        return C[np.triu_indices(C.shape[0], 1)]

    @property
    def max_abs_corr(self) -> float:
        """Largest |correlation| between two distinct clean series."""
        return float(self._abs_corrs().max())

    @property
    def mean_abs_corr(self) -> float:
        """Mean |correlation| over distinct pairs of clean series.

        Reported next to :attr:`max_abs_corr` in every result table because the two
        tell different stories. ``dependence="independent"`` makes the *generating
        processes* independent, but for a strongly autocorrelated base series the
        *realised sample* correlation stays high at moderate T (~0.35 mean / ~0.67 max
        for phase-randomised airpassengers at T=144) — the Yule (1926) nonsense-
        correlation effect, since heavy low-frequency power shrinks the effective
        sample size. Which of the two notions of "independent" the H1/H2 hypotheses
        are about is a live design question, so both are recorded rather than one
        being quietly assumed.
        """
        return float(self._abs_corrs().mean())

    @property
    def signal_rank(self) -> int:
        """Numerical rank of the clean panel (k when shared, p when independent)."""
        s = np.linalg.svd(self.signal, compute_uv=False)
        return int((s > 1e-10 * s[0] * max(self.signal.shape)).sum())


def make_panel(
    T: int = 144,
    p: int = 6,
    k: int = 2,
    *,
    base: str = "synthetic",
    bank_method: str = "surrogate",
    bank_names: Sequence[str] | None = None,
    dependence: str = "shared",
    shared_ratio: float = 0.5,
    noise_sd: float = 0.03,
    contamination: float = 0.0,
    kind: str = "additive",
    magnitude: float = 8.0,
    seed: int | None = 0,
    **contam_kwargs,
) -> Panel:
    """Build a panel from the design-v2 grid: base x dependence x contamination.

    Parameters
    ----------
    T, p, k : length, number of series, number of *common* factors.
    base : ``"synthetic"`` or a real series name (:data:`BASE_SERIES`) used as the
        clean signal / factor bank.
    bank_method : how a real base fills the bank, ``"surrogate"`` (default, independent
        phase-randomised copies) or ``"segment"``; see :func:`factor_bank`.
    bank_names : for ``bank_method="segment"``, which real series to draw from and in
        what order. Matters more than it looks: the second factor sets the panel's
        effective rank, and a noisy series (sunspots) pushes it far above a smooth one
        (co2), which in turn decides how much room a robust fit has to reject outliers.
    dependence : cross-series dependence, one of :data:`DEPENDENCE_LEVELS`.

        * ``shared``      — every series is a random loading combination of the same
          ``k`` factors; the clean panel has rank ``k``. (The Phase-2 setting.)
        * ``independent`` — series ``j`` is driven by its *own* distinct factor and
          nothing else; the clean panel has rank ``p``. This is the regime the
          supervisor's H1/H2 are about.
        * ``partial``     — ``sqrt(shared_ratio)`` of a shared-factor combination plus
          ``sqrt(1 - shared_ratio)`` of an idiosyncratic factor, so dependence is dialled
          continuously between the two extremes.
    shared_ratio : variance share carried by the common factors when ``partial``.
    noise_sd : i.i.d. Gaussian noise sd, in units of the (unit-sd) signal columns.
    contamination, kind, magnitude : passed to :func:`rmssa.contamination.contaminate`.
    **contam_kwargs : type-specific outlier options (``patch_len``, ``phi``, ...).

    Notes
    -----
    Signal columns are standardised to unit sd, so ``noise_sd`` and ``magnitude`` mean
    the same thing for every base series and every dependence level — which is what
    makes cells of the design comparable.
    """
    if dependence not in DEPENDENCE_LEVELS:
        raise ValueError(f"dependence must be one of {DEPENDENCE_LEVELS}, got {dependence!r}")
    if not 0.0 <= shared_ratio <= 1.0:
        raise ValueError(f"shared_ratio must be in [0, 1], got {shared_ratio}")
    rng = np.random.default_rng(seed)

    # how many distinct factors the dependence level needs
    n_needed = {"shared": k, "independent": p, "partial": k + p}[dependence]
    F, labels = factor_bank(T, n_needed, base=base, method=bank_method, rng=rng,
                            names=bank_names)

    # Loadings are kept exactly consistent with the signal at every step, including the
    # rescalings below -- ``signal == factors @ loadings.T`` is a documented part of the
    # ground truth, and a rescaling absorbed into the signal but not the loadings would
    # silently falsify it. Every column rescale is mirrored into the loading rows.
    if dependence == "shared":
        loadings = rng.standard_normal((p, k))
        signal = F @ loadings.T
    elif dependence == "independent":
        loadings = np.eye(p)
        signal = F.copy()
    else:  # partial: shared factors + one idiosyncratic factor per series
        common = rng.standard_normal((p, k))
        # normalise the shared part to unit column sd *through the loadings*, so the
        # shared/idiosyncratic variance split is exactly ``shared_ratio``
        shared_sd = (F[:, :k] @ common.T).std(axis=0)
        common = common / np.where(shared_sd > 0, shared_sd, 1.0)[:, None]
        w = np.sqrt(shared_ratio)
        loadings = np.hstack([w * common, np.sqrt(1 - shared_ratio) * np.eye(p)])
        signal = F[:, : k + p] @ loadings.T

    # final per-column standardisation, mirrored into the loadings
    signal_sd = signal.std(axis=0)
    signal_sd = np.where(signal_sd > 0, signal_sd, 1.0)
    signal = _column_standardise(signal)
    loadings = loadings / signal_sd[:, None]
    noise = noise_sd * rng.standard_normal((T, p))

    contam = contaminate(
        signal, kind=kind, rate=contamination, magnitude=magnitude, rng=rng, **contam_kwargs
    )

    return Panel(
        X=signal + noise + contam.outliers,
        signal=signal,
        factors=F,
        loadings=loadings,
        noise=noise,
        outliers=contam.outliers,
        mask=contam.mask,
        base=base,
        dependence=dependence,
        kind=kind,
        factor_labels=labels,
        meta={
            "T": T, "p": p, "k": k, "shared_ratio": shared_ratio,
            "noise_sd": noise_sd, "seed": seed, "bank_method": bank_method,
            **contam.meta,
        },
    )


def _column_standardise(M: np.ndarray) -> np.ndarray:
    """Each column to zero mean / unit sd (constant columns left alone)."""
    M = np.asarray(M, dtype=float)
    sd = M.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (M - M.mean(axis=0)) / sd


# ------------------------------------------------------------------------------ yahoo
def _impersonating_session():
    """A curl_cffi browser-impersonating session, or None if curl_cffi is absent.

    Yahoo aggressively rate-limits (HTTP 429) requests that don't look like a real
    browser. A curl_cffi session impersonating Chrome dodges most of that; if curl_cffi
    isn't installed we return None and let yfinance use its own default session (which,
    on a recent yfinance, is usually sufficient).
    """
    try:
        from curl_cffi import requests as _creq

        return _creq.Session(impersonate="chrome")
    except Exception:
        return None


def load_yahoo(
    tickers: Sequence[str] | None = None,
    start: str = "2005-01-01",
    end: str = "2024-12-31",
    *,
    field: str = "Close",
    returns: bool = True,
    cache_dir: str | Path = "data/raw",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download (and cache) daily prices for an index panel; optionally to returns.

    Tries the on-disk cache first, then yfinance. Raises a clear RuntimeError if the
    data is neither cached nor downloadable (e.g. offline or rate-limited), so callers
    can fall back to :func:`make_synthetic_panel`.

    Returns a wide DataFrame indexed by date, one column per ticker, NA rows dropped.
    """
    tickers = list(tickers) if tickers is not None else list(DEFAULT_INDICES)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(t.replace("^", "") for t in tickers)
    cache_file = cache_dir / f"yahoo_{tag}_{start}_{end}_{field}.csv"

    prices: pd.DataFrame | None = None
    if use_cache and cache_file.exists():
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)

    if prices is None:
        try:
            import yfinance as yf  # local import: optional dependency

            raw = yf.download(
                tickers, start=start, end=end, auto_adjust=True, progress=False,
                session=_impersonating_session(),  # None if curl_cffi unavailable
            )
            if raw is None or raw.empty:
                raise RuntimeError(
                    "yfinance returned no data. Most common cause is an outdated "
                    "yfinance (Yahoo now rejects old request patterns with HTTP 429); "
                    "run `pip install -U yfinance` (>=1.4). If still blocked, the IP may "
                    "be rate-limited — wait, use a different network, or install curl_cffi."
                )
            prices = raw[field] if field in raw.columns.get_level_values(0) else raw["Close"]
            prices = prices.dropna(how="all")
            if use_cache:
                prices.to_csv(cache_file)
        except Exception as exc:  # network / rate-limit / dependency
            raise RuntimeError(
                f"Could not load Yahoo data for {tickers} (no cache at {cache_file}). "
                f"Underlying error: {exc!r}. Fall back to make_synthetic_panel()."
            ) from exc

    prices = prices.sort_index().dropna()
    if returns:
        out = np.log(prices).diff().dropna()
        return out
    return prices
