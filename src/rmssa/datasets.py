"""Dataset loaders and synthetic panel generators.

Day-8 scope: a small real equity-index loader (Yahoo Finance, with on-disk caching and
a graceful offline fallback) and a synthetic panel generator with a *known* low-rank
factor structure and injectable outliers (used heavily in Phase-2 validation).

The synthetic generator returns ground truth (factors, loadings, clean signal, outlier
mask) so that subspace-recovery error can be computed exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["SyntheticPanel", "make_synthetic_panel", "load_yahoo", "DEFAULT_INDICES"]


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


# ------------------------------------------------------------------------------ yahoo
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
                tickers, start=start, end=end, auto_adjust=True, progress=False
            )
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned no data")
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
