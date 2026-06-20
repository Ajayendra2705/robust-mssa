# Robust SVD / Robust Low-Rank Estimators — Comparison (Day 3)

> Purpose: seed the **methodological discussion with the group** — *which robust SVD variant
> is most appropriate for the MSSA trajectory-matrix setting?* (explicit open question in the
> proposal). The trajectory matrix `H ∈ R^{L×K_tot}` is **structured** (block-Hankel) and
> **wide** (`K_tot = pK ≫ L`), with outliers appearing as **a few corrupted columns**
> (an outlier date contaminates an entire diagonal band of `H`, i.e. several columns), not as
> i.i.d. cell-wise corruption. The right estimator must respect that geometry.
>
> Items marked **[verify]** must be confirmed against Kazemi & Rodrigues (2023) and
> Neto & Rodrigues (2022) before committing.

---

## Why standard SVD fails here

Standard SVD solves `min_{rank(M)≤r} ‖H − M‖_F²` (L2 / Eckart–Young). The squared
Frobenius norm gives outliers quadratic leverage: a small number of corrupted columns can
inflate the residual and **rotate the leading left singular vectors** `U_i` away from the
true signal subspace. Since in MSSA the `U_i` are the *shared temporal factors* we care
about, this rotation directly corrupts the extracted factors and everything downstream.

The robustness goal: estimate the rank-`r` column/row subspace of the *clean* signal `S`
while down-weighting or isolating the outlier contribution `O` in `H = S + N + O`.

---

## Candidate estimators

| # | Method | Objective / mechanism | Outlier model it suits | Cost (per iter) | Hyperparameters | Notes for MSSA |
|---|--------|-----------------------|------------------------|-----------------|-----------------|----------------|
| 1 | **IRLS / M-estimator weighted SVD** (Huber, Tukey bisquare) | Iteratively reweighted least squares: `min Σ ρ(r_{ij})`; reweight rows/cols/cells by residual, re-run weighted SVD | row-, column-, or cell-wise, moderate fraction | one SVD per iter `O(L·K_tot·r)` | tuning const `c` (e.g. Huber 1.345σ), scale estimate (MAD) | **Column-weighted** form maps cleanly onto "outlier date = bad column"; simplest drop-in behind `decompose`; natural **default for Day 13** |
| 2 | **Robust PCA / PCP** (Candès et al. 2011) | `min ‖L‖_* + λ‖S‖_1 s.t. H = L + S` (nuclear + L1, convex) | **sparse cell-wise** gross corruption | SVD + soft-threshold per iter (ADMM/IALM), `O(L·K_tot·min(L,K_tot))` | `λ ≈ 1/√max(L,K_tot)`, tol | Strong guarantees but assumes *sparse* `S`; column-clustered outliers violate sparsity → consider **block/column RPCA** variant |
| 3 | **L1-norm low-rank approx** (Ke & Kanade 2005; Eriksson & van den Hengel) | `min_{rank(M)≤r} ‖H − M‖_1` via alternating convex / weighted median | heavy-tailed, cell-wise | LP/alternating, can be slow on wide `H` | rank `r`, tol | More robust than L2; scalability on `K_tot = pK` is a concern (Day 22) |
| 4 | **RobRSVD — regularized robust SVD** (Zhang, Shen & Huang 2013) | robust loss (e.g. Huber) **+** smoothing penalties on singular vectors, rank-1 deflation | contaminated + smooth singular vectors | per-component alternating, moderate | robust scale, smoothing params | Smoothing on `U_i` is attractive for time-series factors; deflation fits the eigentriple view |
| 5 | **Outlier-robust / column-wise (R1-PCA, spherical/sign PCA)** | project to unit sphere or use spatial sign; or detect & down-weight whole columns | **column-wise** outliers | one SVD, cheap | none/few | Directly targets "bad column" geometry of `H`; cheap; good diagnostic baseline |
| 6 | **Randomized robust SVD** | randomized range finder + robust core (init for #1–#4) | speed, long series | sub-SVD `O(L·K_tot·k)` | oversampling `p`, power iters `q` | Pairs with Rodrigues, Tuy & Mahmoudvand (2018) randomized SSA → Day-22 speedups for long `T` |
| 7 | **Kernel robust SVD** (Neto & Rodrigues 2022) | robust SVD in an RKHS feature space (non-linear) | non-linear structure + contamination | kernel matrix `O(K_tot²)` build | kernel + bandwidth, robust params | Phase-2 **second variant (Day 19)**; tests modularity + captures non-linear co-movement; heavier |

---

## Decision criteria (to settle with supervisor)

1. **Outlier geometry.** Confirm the dominant contamination mode. In a Hankel/block-Hankel
   `H`, a single outlier observation in series `j` appears in up to `L` columns of block `j`
   along an anti-diagonal → **column/block-structured**, not i.i.d. sparse. This favours
   **column-weighted IRLS (#1)** and **column-wise/sign methods (#5)** over vanilla PCP (#2),
   *unless* a Hankel-aware RPCA is used. **[verify which mode Kazemi & Rodrigues target]**
2. **Which estimator(s) Kazemi & Rodrigues (2023) actually evaluate** — adopt their variant as
   the primary so the work is a direct extension, not an import. **[verify]**
3. **Breakdown point vs cost.** Higher-breakdown estimators (L1, high-breakdown M) cost more;
   `K_tot = pK` can be large. Need a variant that scales to the empirical panels (Phase 3).
4. **Interface fit.** All must implement the same `decompose(H, r) → (U, s, Vt)` contract so
   the modularity claim holds and they are directly comparable.

## Recommendation for the build (pending the discussion)
- **Day 13 default:** **column-weighted IRLS (Huber/bisquare) SVD (#1)** — simplest faithful
  robustification, matches the column-outlier geometry, cheap, easy to validate against
  standard SVD at ε = 0.
- **Day 19 second variant:** **kernel robust SVD (#5/#7, Neto & Rodrigues 2022)** — stress-tests
  modularity and adds a non-linear comparator.
- Keep **RobRSVD (#4)** and **column RPCA (#2-variant)** as ranked alternates if the group
  prefers a different primary.

---

## Open questions to email the group (end of Day 10 checkpoint)
1. Which robust SVD variant(s) does Kazemi & Rodrigues (2023) use, and should that be our
   primary estimator for direct comparability?
2. Do you view the trajectory-matrix outliers as column-structured (date-driven) or cell-wise?
   This determines whether PCP-style sparsity assumptions are appropriate.
3. Is a **Hankel-structure-preserving** robust low-rank step of interest (project to Hankel
   each iteration), or is unstructured robust SVD + diagonal averaging sufficient?
4. For the kernel robust variant — recommended kernel/bandwidth selection for financial panels?
