# Robust MSSA

**Robust Multivariate Singular Spectrum Analysis for Latent Factor Extraction in Cross-Sectional Financial Time Series**

Research internship project · SaLLy (Statistical Learning Laboratory), Federal University of Bahia
Intern: Ajayendra Kumar Bansod (IIT Kharagpur) · Supervisor: Prof. Paulo Canas Rodrigues

---

## Motivation

A panel of `p` financial time series observed over `T` points,
`X = [x_1, …, x_p] ∈ R^{T×p}`, can be modelled as

```
X = S + N + O
```

where **S** is a low-rank shared signal (trend, cycle, cross-sectional co-movement),
**N** is idiosyncratic noise, and **O** is sparse outlier contamination from market
events (flash crashes, earnings shocks, macro announcements).

- **Standard MSSA** (Rodrigues & Mahmoudvand, 2018) recovers the shared structure `S`
  but its L2/SVD step is **non-robust** — a few outlier columns rotate the leading
  singular vectors away from the true signal subspace.
- **Robust SSA** (Rodrigues et al., 2020; Kazemi & Rodrigues, 2023) is robust to outliers
  but is **univariate**, discarding cross-sectional structure.

This project combines them: **Robust MSSA** replaces the SVD step inside the MSSA
pipeline with a robust low-rank estimator, and tests whether this yields more stable,
interpretable factor decompositions on contaminated financial panels — and whether the
gain translates to out-of-sample evaluation.

## Central research question

> Can replacing the standard SVD step in MSSA with a robust SVD yield more stable and
> interpretable factor decompositions on contaminated financial panels, and does this
> translate into improved out-of-sample evaluation of the extracted components?

Tested with a **2×2 factorial** across datasets and contamination regimes —
**{classical, robust} × {univariate SSA, multivariate MSSA}** — so the two research
axes (classical-vs-robust and univariate-vs-multivariate) are cleanly separable. The
two robust SVD algorithms are those of **Rodrigues, Pimentel, Messala & Kazemi (2020,
*Entropy*)**: the Huber (**RHSSA**) and L1-norm (**RLSSA**) variants.

## Design commitment: the SVD step is interchangeable

The decomposition backend is a pluggable component. Embedding, grouping,
reconstruction, and forecasting do **not** depend on which backend is used:

```python
from rmssa.decomposition import StandardSVD          # classical L2 SVD (baseline)
from rmssa.decomposition import RobRSVD               # RHSSA — Huber robust SVD
from rmssa.decomposition import AlternatingL1SVD      # RLSSA — L1-norm robust SVD

from rmssa import MSSA
model = MSSA(window=50, backend=RobRSVD(rank=6)).fit(panel)   # robust MSSA
```

All three implement one contract, `decompose(H) -> (U, s, Vt)`; the robust backends
collapse to the ordinary SVD when the data is clean, so classical-vs-robust is a fair
comparison at zero contamination.

## Status

| Phase | Days | State |
|-------|------|-------|
| 1 — Foundations & standard-MSSA baseline | 1–10 | **done** (`v0.1-baseline`) — reference-validated |
| 2 — Robust MSSA + synthetic validation | 11–20 | **done** (`v0.2-robust-synthetic`) — 99 tests passing |
| 2b — Hardening & wrap | 21–25 | in progress |
| 3 — Empirical study (equity + macro) | 26–40 | pending |
| 4 — Out-of-sample evaluation | 41–50 | pending |
| 5 — Report, release, paper draft | 51–60 | pending |

## Results so far

Synthetic validation against a known clean signal (`report/phase2_summary.md` ties these
together):

- **No robustness tax at ε=0; robust wins under contamination** — [`results_phase2_grid.md`](report/results_phase2_grid.md).
- **Contamination sweep** (recovery + subspace error vs ε) — [`results_phase2_sweep.md`](report/results_phase2_sweep.md).
- **Where the gain is largest/smallest** (vary k/p/T/L) — [`results_phase2_dimsweep.md`](report/results_phase2_dimsweep.md).
- **Huber vs L1** (near-equivalent; Huber cheaper) — [`results_phase2_algocompare.md`](report/results_phase2_algocompare.md).

Headline: at 2% contamination Robust MSSA (Huber, multivariate) recovers the clean signal
~27× more accurately than classical MSSA; classical collapses as contamination rises while
Robust MSSA degrades gracefully. Every figure/number regenerates from a script + config + seed.

## Install

```bash
# editable install with dev + data extras
python -m pip install -e ".[dev,data]"
# or with conda
conda env create -f environment.yml && conda activate rmssa
```

## Run tests

```bash
pytest                 # fast suite (no network, no heavy deps)
pytest -m external     # + cross-validation against pyts (slower)
```

Reference validation (analytic SSA-rank ground truth + pyts cross-check) is also runnable
as a standalone report:

```bash
python experiments/01_baseline_repro/validate_reference.py   # -> report/validation_reference.md
```

## Troubleshooting: Yahoo Finance HTTP 429 / "possibly delisted; No timezone found"

`yfinance` pulls from Yahoo's unofficial endpoints, which rate-limit (HTTP 429) requests
that don't look like a real browser. A 429 during the timezone pre-fetch surfaces
confusingly as `YFTzMissingError('… possibly delisted; No timezone found')` — the ticker is
fine; the request was throttled. In order of effectiveness:

1. **Upgrade yfinance** — by far the most common cause. Old versions (e.g. 0.2.40) use a
   request pattern Yahoo now blocks. `pip install -U yfinance` (>=1.4) fixed it here.
2. **Use a browser-impersonating session** — `load_yahoo` automatically uses a
   `curl_cffi` Chrome session if `curl_cffi` is installed (`pip install curl_cffi`).
3. **Back off / cache** — the loader caches to `data/raw/` on first success, so subsequent
   runs don't re-hit Yahoo. Avoid tight download loops; add delays for many tickers.
4. **Different network** — datacenter/CI IP ranges are sometimes blocked outright; a 429
   that survives 1–3 usually means the IP itself is throttled. Wait, or use another network.
5. **Offline fallback** — `make_synthetic_panel()` lets the whole pipeline run without Yahoo.

## Layout

```
src/rmssa/      core library: embedding, decomposition (StandardSVD / RobRSVD / AlternatingL1SVD),
                grouping, reconstruction, mssa, metrics, datasets  (forecast: Phase 4)
experiments/    reproducible studies, one folder per phase; configs in experiments/configs;
                experiment outputs (figures/CSVs) are gitignored — regenerate from the scripts
tests/          pytest suite (embedding & reconstruction identities, backend equivalence,
                robust correctness + independent cross-check, metrics)
report/         literature notes, robust-SVD comparison, per-phase result notes, checkpoint summary
```

Regenerate the synthetic-validation results:

```bash
python experiments/02_synthetic_validation/run_grid.py     --config experiments/configs/grid_synthetic.yaml
python experiments/02_synthetic_validation/run_sweep.py    --config experiments/configs/sweep_synthetic.yaml
python experiments/02_synthetic_validation/run_dimsweep.py --config experiments/configs/dimsweep_synthetic.yaml
python experiments/02_synthetic_validation/run_algo_compare.py --config experiments/configs/algo_compare.yaml
```

## References

- Rodrigues, P.C. & Mahmoudvand, R. (2018). *The benefits of multivariate singular spectrum analysis over the univariate version.* J. Franklin Institute.
- **Rodrigues, P.C., Pimentel, J., Messala, P. & Kazemi, M. (2020). *The decomposition and forecasting of mutual investment funds using singular spectrum analysis.* Entropy 22(1):8.** — source of the two robust SVD algorithms (RHSSA, RLSSA).
- Hawkins, D.M., Liu, L. & Young, S. (2001). *Robust singular value decomposition.* NISS. — the L1-norm robust SVD (RLSSA).
- Zhang, L., Shen, H. & Huang, J.Z. (2013). *Robust regularized singular value decomposition with application to mortality data.* Ann. Appl. Stat. — the Huber robust SVD (RHSSA).
- Kazemi, M. & Rodrigues, P.C. (2023). *Robust singular spectrum analysis: comparison between classic and robust approaches.* Computational Statistics.
- Neto, E.A.L. & Rodrigues, P.C. (2022). *Kernel robust singular value decomposition.* Expert Systems with Applications.
- Rodrigues, P.C. & Mahmoudvand, R. (2020). *A new approach for the vector forecast algorithm in SSA.* Comm. Stat. Sim. Comp.
- de Carvalho, M., Rodrigues, P.C. & Rua, A. (2012). *Tracking the US business cycle with a singular spectrum analysis.* Economics Letters.

## License

MIT — see [LICENSE](LICENSE).
