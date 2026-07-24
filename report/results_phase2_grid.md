# Phase 2 — Day 16: first full-grid comparison on synthetic panels

**Experiment:** `experiments/02_synthetic_validation/run_grid.py`
**Config:** `experiments/configs/grid_synthetic.yaml` — T=300, p=6, k=2 shared factors,
noise_sd=0.03, L=50, r=6, 5 seeds, contamination ε ∈ {0, 1%, 2%}.
**Metric:** signal-recovery error = ‖reconstructed signal − clean S‖_F / ‖S‖_F (lower = better).

## The 2×2 factorial

{classical (StandardSVD), RHSSA (Huber robust SVD), RLSSA (L1 robust SVD)}
× {univariate SSA per series, multivariate MSSA}.

## Mean recovery error over 5 seeds

| config | ε = 0 | ε = 1% | ε = 2% |
|---|---|---|---|
| classical · univariate   | 0.010 | 0.318 | 0.415 |
| classical · multivariate | 0.010 | 0.265 | 0.347 |
| **RHSSA (Huber) · multivariate** | 0.010 | **0.016** | **0.013** |
| RLSSA (L1) · multivariate | 0.010 | 0.028 | 0.018 |
| RHSSA (Huber) · univariate | 0.010 | 0.089 | 0.046 |
| RLSSA (L1) · univariate | 0.010 | 0.104 | 0.059 |

![recovery vs contamination](../experiments/02_synthetic_validation/outputs/grid_recovery.png)

## Findings

1. **No robustness tax at ε = 0.** All six configs recover the clean signal to ~0.010
   (the ~1% noise floor). Classical vs robust is therefore a *fair* comparison — the
   robust backends collapse to the ordinary SVD when there is nothing to be robust to.
2. **Classical SSA/MSSA collapse under contamination** (0.010 → 0.35–0.42 at ε = 2%):
   a handful of outliers rotate the L2 singular subspace and corrupt the extracted
   signal.
3. **Robust methods stay accurate.** Robust error barely moves with ε.
   **Robust MSSA (Huber · multivariate) is the best config** (0.013 at ε = 2%), a
   **~27×** lower error than classical MSSA there — the project's target method wins.
4. **Multivariate beats univariate** for every method (observations all negative:
   MSSA − SSA ≈ −0.03 to −0.07 at ε = 2%): sharing structure across the panel both
   improves recovery and makes the robust fit more stable, since an outlier date
   corrupts a smaller share of the wider block-Hankel matrix.
5. Huber (RHSSA) edges out L1 (RLSSA) on this Gaussian-noise setup; the gap should
   narrow (or reverse) under heavier-tailed noise — to be probed in the Day-17 sweep.

## Notes / caveats

- **Rank matters.** r must be ≥ the signal's SSA-rank (here 6: factor 0 = sinusoid +
  linear trend → rank ~4, factor 1 = sinusoid → rank 2). At r = 4 every config left a
  ~20% residual even at ε = 0, and the robust fit then mistook unmodelled signal for
  outliers. Rank selection is a first-class hyperparameter for Day 21.
- **Solver caps.** The robust backends run with `max_iter=60, tol=1e-6` here (see
  config `solver:`); verified to match the fully-converged (`tol=1e-9`) error within
  ~1e-3 while cutting runtime ~4×. Library defaults remain the tighter `200 / 1e-9`.
- All numbers reproduce from `python experiments/02_synthetic_validation/run_grid.py`
  with the committed config + seeds.

**Next (Day 17):** widen to ε ∈ {1%, 5%, 10%, 20%}, add subspace-recovery error
alongside reconstruction error, and vary k / p / T / L.
