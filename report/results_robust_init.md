# The initialisation basin: why the robust fits were losing, and what fixed it

The R cross-check (`results_rcheck.md`) found that both robust backends diverged from the
R references on a narrow 40×42 trajectory matrix, and traced it to the solver being an
IRLS *by imputation*: down-weighted cells are replaced by the **current model's own
values**, so the current model is a fixed point and a start already captured by the
outliers cannot be escaped. This note follows that through, and reports two things that
the original diagnosis got wrong.

## 1. The failure is not confined to narrow matrices

It is a *starting point* failure, and narrowness only makes it more likely. Scoring against
the clean-signal subspace (r = 2, 10% of cells contaminated at 15× sd, 10 seeds):

| fixture | K | median distance to truth | seeds failing (> 0.25) |
|---|---:|---:|---:|
| narrow | 42 | 0.6371 | 6 / 10 |
| medium | 82 | 0.0774 | 2 / 10 |
| wide | 273 | 0.0634 | 1 / 10 |
| mssa-scale | 805 | 0.0471 | 2 / 10 |

Two of ten seeds fail at K = 805 — the realistic MSSA size, and the size at which the R
cross-check *passed*. The cross-check used a single seed and happened to draw a good one.
The claim in `results_rcheck.md` that the validated region is "K ≥ 122" was therefore too
generous: width shifts the *rate* of capture, it does not remove it.

## 2. The fix: run both starts, keep the one with the lower objective

Neither candidate start is safe on its own, which is the part worth stating carefully:

* The **classical SVD of H** is already rotated onto the outliers whenever they dominate.
* A **Winsorized start** (column-wise clipping at median ± 2.5 MAD) escapes that basin —
  but it perturbs an exactly low-rank *clean* matrix off its exact solution, and the
  iteration sticks there too. Used alone it breaks the property the whole comparison rests
  on: that at ε = 0 the robust backend reproduces the classical SVD. It did break it: the
  clean-data equivalence test in `test_robust_decomposition.py` failed under a
  Winsorized-only start.

So the default (`init="auto"`) runs the iteration from both and keeps whichever fit has the
lower M-estimation objective, `sum rho(r_ij / sigma)`, with rho the same function the
backend's weights are the IRLS weights of (Huber at c = 1.345 for RHSSA, c = 1 for RLSSA).
The scale `sigma` is **fixed across candidates** — this matters, because a fit that has
absorbed the outliers has small residuals *at those cells* and would shrink its own MAD
scale and win a scale-free comparison. Scored at a common scale, the clean-data classical
fit has residual and objective exactly zero and always wins; under heavy contamination the
Winsorized fit wins. The estimator, its weights and its fixed-point equation are untouched;
only the choice of which fixed point is reported changes. Cost is one extra IRLS run
(1.78 s → 3.46 s per decomposition on a 300×6 panel at L = 50).

## 3. What it recovers

**Narrow fixtures**, same setting as the table above:

| fixture | K | old (classical start) | new (both starts) | failures old → new |
|---|---:|---:|---:|---|
| narrow | 42 | 0.6371 | **0.0757** | 6/10 → 1/10 |
| medium | 82 | 0.0774 | **0.0663** | 2/10 → 1/10 |
| wide | 273 | 0.0634 | 0.0634 | 1/10 → 1/10 |
| mssa-scale | 805 | 0.0471 | 0.0471 | 2/10 → 2/10 |

**The Phase-2 grid**, which was supposed to be inside the validated region — signal
recovery error on the trajectory matrix, T = 300, p = 6, L = 50, r = 6, 5 seeds:

| ε | old | new | change |
|---:|---:|---:|---|
| 0% | 0.01073 | 0.01073 | unchanged to 3e-12 |
| 1% | 0.02383 | 0.01327 | 1.8× better |
| 2% | 0.01467 | 0.01122 | 1.3× better |
| 5% | 0.04992 | 0.01667 | 3.0× better |
| 10% | 0.08603 | 0.03737 | 2.3× better |
| 20% | 0.21064 | 0.18122 | 1.2× better |

ε = 0 is unchanged to twelve digits, which is the check that the fix has not quietly made
the robust and classical arms different estimators on clean data.

**The seed spread was mostly this, not sampling noise.** The gain over classical MSSA per
seed (10 seeds), before and after:

| ε | old per-seed gain | new per-seed gain | old median → new | old spread → new |
|---:|---|---|---|---|
| 2% | 7.0 … 38.5 | 32.8 … 38.7 | 34.5× → 35.8× | 5.5× → **1.2×** |
| 5% | 6.2 … 50.5 | 25.6 … 57.0 | 25.9× → **49.8×** | 8.1× → **2.2×** |
| 10% | 6.0 … 51.7 | 9.7 … 64.8 | 15.5× → **49.8×** | 8.7× → 6.7× |

The low seeds — 7.0, 6.2, 6.0 — were not hard draws of the data. They were the optimiser
being captured. This directly revises the caveat carried in the design-v2 write-up, that
the additive multiplier "is not pinned down by 3 seeds (per-seed 25.6/16.4/4.0)": a good
part of that scatter was the solver, and it largely disappears. The *ordering* of
contamination types, which was the actual finding there, is unaffected — but the
contamination-type and forecasting tables should be re-run before anything is quoted.

## 4. A separate correction: "fair at ε = 0" needs a rank condition

The module comment claimed that on clean data residuals vanish, weights go to 1 and the fit
collapses to the ordinary SVD, so classical vs robust is a fair comparison at ε = 0. That
holds only when **r ≥ rank(signal)**. Below it the residual is not noise but *discarded
signal*; the MAD scale is set by that signal, and the Huber weight down-weights legitimate
structure. Noise-free, no contamination at all, on a panel of exact rank 6:

| r | distance between robust and classical subspaces | robust error / L2-optimal error |
|---:|---:|---:|
| 1 | 0.30 | 1.004 |
| 2 | 0.38 | 1.010 |
| 3 | **0.57** | 1.021 |
| 4 | 0.059 | 1.047 |
| 5 | 0.014 | 1.237 |
| 6 | 3.3e-08 | 1.000 |
| 8 | 3.3e-08 | 1.000 |

Both backends behave the same way. The practical reading: the *approximation* stays within
1–5% of the L2 optimum at every rank, so signal-recovery comparisons remain fair — but
*subspace* comparisons below the signal rank do not, and the R cross-check ran every
fixture at r = 2 against a rank-6 signal, i.e. deep inside that regime.

One more metric caveat found alongside it: `subspace_distance` is the sine of the
**largest** principal angle, so a single unrecoverable weak direction saturates it. At
r = 6 on the contaminated fixtures it reports ≈ 1.00 for the robust *and* the classical fit
while their reconstruction errors differ by 1.3–3.0×; mean-cos² overlap separates them
cleanly (0.83 vs 0.38). Largest-angle distance should not be used as the headline accuracy
metric once r reaches into the weak part of the spectrum.

## Reproducing

`init="classical"` restores the Phase-2 behaviour exactly, so every table above can be
regenerated by flipping that one argument. Tests covering the fix:
`test_default_init_escapes_the_outlier_basin`,
`test_default_init_keeps_exactness_on_clean_low_rank_data`.
