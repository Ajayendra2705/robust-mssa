# Phase 2 — Day 19: Huber (RHSSA) vs L1 (RLSSA) robust algorithms

**Experiment:** `experiments/02_synthetic_validation/run_algo_compare.py`
**Config:** `experiments/configs/algo_compare.yaml` — baseline panel (T=300, p=6, k=2,
L=50, r=6), ε ∈ {0, 1%, 5%, 10%, 20%}, 5 seeds, multivariate. Solver caps raised to
`max_iter=300, tol=1e-8` so convergence (`n_iter_`) is measured honestly.

![Huber vs L1](../experiments/02_synthetic_validation/outputs_algocompare/algo_compare.png)

## Results (mean over 5 seeds)

| ε | inter-algo divergence | Huber err | L1 err | Huber iters | L1 iters | Huber s | L1 s |
|---|---|---|---|---|---|---|---|
| 0    | 0.0009 | 0.0100 | 0.0103 | 34  | 57  | 0.88 | 1.48 |
| 1%   | 0.0010 | 0.0102 | 0.0105 | 162 | 204 | 4.38 | 5.58 |
| 5%   | 0.0016 | 0.0112 | 0.0113 | 209 | 241 | 4.34 | 4.96 |
| 10%  | 0.0033 | 0.0613 | 0.0601 | 272 | 286 | 5.17 | 5.56 |
| 20%  | 0.0053 | 0.1468 | 0.1525 | 271 | 300 | 6.36 | 7.11 |

*divergence = sin of the largest principal angle between the two algorithms' leading
factor subspaces (0 = identical).*

All 3 checks pass: agree at ε=0, both recover the clean signal, divergence grows with ε.

## Findings

1. **The two algorithms are near-equivalent in accuracy.** Their leading factor
   subspaces differ by < 0.006 (largest principal angle) even at ε = 20%, and recovery
   errors track within ~5%. They share one IRLS-by-imputation engine and differ only in
   the weight function (Huber min(1, c·s/|r|), c=1.345, vs L1 min(1, s/|r|)), so the
   estimates stay close. Neither dominates: Huber is marginally better at ε ≤ 5% and at
   20%, L1 a hair better at 10%.
2. **Agreement at ε=0 confirmed** (divergence 0.0009): both reduce to the ordinary SVD
   when there is nothing to down-weight.
3. **Divergence grows monotonically with ε** (0.0009 → 0.0053) — the weight functions
   part company more as more residuals cross their thresholds — but the gap stays small.
4. **Huber is cheaper.** It converges in fewer IRLS sweeps at every ε (e.g. 34 vs 57 at
   ε=0) and runs faster in wall time. Cost climbs steeply with contamination for both:
   from ~35 iters clean to ~270–300 at ε ≥ 10%.
5. **Convergence caveat.** At ε ≥ 10% both approach the 300-iteration cap and L1 hits it
   at ε=20% (mean 300) — i.e. neither fully meets tol=1e-8 under heavy contamination.
   Reported errors there are at the cap; a looser tol (as the sweeps use) reaches
   essentially the same accuracy far sooner.

## Recommendation

**Default to Huber (RHSSA).** It matches L1's accuracy while converging faster and in
fewer iterations. Keep L1 (RLSSA) as the reported second algorithm (per the supervisor's
two-algorithm directive) and as a heavier-tailed comparator for the empirical phase,
where non-Gaussian noise may favour its more aggressive tail down-weighting.

## Notes

- `RobustSVD` now exposes `n_iter_` and `converged_` after `decompose` (new diagnostic,
  unit-tested).
- Reproduce: `python experiments/02_synthetic_validation/run_algo_compare.py
  --config experiments/configs/algo_compare.yaml`.

**Next (Day 20):** mid-internship checkpoint — package the synthetic findings (Days 16–19)
for the supervisor; tag `v0.2-robust-synthetic`.
