# Literature Review (Days 2–4)

**Project:** Robust MSSA for latent factor extraction in cross-sectional financial time series
**Author:** Ajayendra Kumar Bansod · **Status:** Phase 1 deliverable

This note synthesises the methodological foundations the project builds on, drawn entirely
from the supervisor's group's contributions, plus the canonical SSA references. Companion
notes: [`ssa_math_notes.md`](ssa_math_notes.md) (mechanics) and
[`robust_svd_comparison.md`](robust_svd_comparison.md) (estimator options). **[verify]** tags
mark claims to confirm against the source papers before they enter the technical report.

---

## 1. The gap this project fills

Two strands of the group's work meet at an unexplored intersection:

- **Multivariate SSA** — *Rodrigues & Mahmoudvand (2018), "The benefits of MSSA over the
  univariate version", J. Franklin Inst.* — establishes when and why a **shared temporal basis**
  across series outperforms `p` independent univariate SSAs. Mechanism: the common left
  singular vectors of the stacked trajectory matrix pool cross-sectional co-movement, so shared
  latent structure is estimated from `pK` columns rather than `K`. **[verify exact conditions /
  metrics under which the benefit is proven]**

- **Robust SSA** — *Kazemi & Rodrigues (2023), "Robust SSA: comparison between classic and
  robust approaches", Computational Statistics* — replaces the non-robust L2 SVD with robust
  low-rank estimation so that outliers (which the squared Frobenius objective gives quadratic
  leverage) do not rotate the leading singular vectors off the signal subspace. Evaluated for
  both **model fit and forecasting**. **[verify which robust estimators are used and the headline
  result]**

**MSSA is multivariate but non-robust; Robust SSA is robust but univariate.** Financial panels
need both at once: assets share macro drivers (→ multivariate) *and* are routinely hit by
outlier events — flash crashes, earnings shocks, macro announcements (→ robust). **Robust
MSSA** = inserting the robust low-rank step (Kazemi & Rodrigues 2023; kernel variant Neto &
Rodrigues 2022) into the MSSA pipeline (Rodrigues & Mahmoudvand 2018). The combination is
methodologically motivated, not ad hoc, and has not been studied on contaminated financial
panels.

### Problem statement
`X = S + N + O` for a panel `X ∈ R^{T×p}`: `S` low-rank shared signal (trend, cycle,
co-movement), `N` idiosyncratic noise, `O` sparse outlier contamination. Recover the subspace
of `S` from the block trajectory matrix `H ∈ R^{L×K_tot}` while resisting `O`.

### Research question
> Can replacing the standard SVD step in MSSA with a robust SVD yield more stable and
> interpretable factor decompositions on contaminated financial panels, and does the gain
> translate into improved out-of-sample evaluation of the extracted components?

Tested via three comparators: **standard MSSA** / **column-wise Robust SSA** (robust but
univariate) / **Robust MSSA** (proposed), across datasets and contamination rates
ε ∈ {0, 1%, 5%, 10%, 20%}.

---

## 2. SSA / MSSA mechanics (summary)

Four stages — full derivation in `ssa_math_notes.md`:
1. **Embedding** → (block-)Hankel trajectory matrix `X ∈ R^{L×K}` (`H ∈ R^{L×K_tot}` for MSSA).
2. **Decomposition** → SVD into eigentriples `(√λ_i, U_i, V_i)`; **this is the swappable step.**
3. **Grouping** → assign eigentriples to trend / oscillatory pairs / noise (w-correlation guides).
4. **Reconstruction** → diagonal averaging (Hankelization) back to component series.

Key diagnostics: **scree plot** (`λ_i / Σλ_j`), **w-correlation matrix** (separability),
phase portraits of paired singular vectors (periodic components). MSSA uses the horizontal
(common-window) stacking so the left singular vectors are shared across channels.

---

## 3. Robustness — why and how

Standard SVD = `min_{rank≤r} ‖H − M‖_F²`; outliers get quadratic leverage and rotate the
leading `U_i`. Robust alternatives replace the L2 loss (IRLS/M-estimators, L1 low-rank,
robust PCA/PCP, RobRSVD) or operate in a feature space (kernel robust SVD). The MSSA twist:
`H` is **block-Hankel and wide** (`K_tot = pK ≫ L`), and a single bad observation contaminates
a band of columns → outliers are **column/block-structured**, which steers estimator choice.
Full table and recommendation in `robust_svd_comparison.md`. Planned: column-weighted IRLS as
the Day-13 default; kernel robust SVD as the Day-19 second variant; **final choice deferred to
the supervisor discussion** as the proposal requires.

---

## 4. Forecasting & evaluation foundations

- **Vector forecasting in SSA** — *Rodrigues & Mahmoudvand (2020), "A new approach for the
  vector forecast algorithm in SSA", Comm. Stat. Sim. Comp.* Two standard continuations:
  - **Recurrent (R-) forecast:** a linear recurrence relation (LRR) derived from the leading
    left singular vectors. With `U_i = (U_i^∇; π_i)` (last coordinate `π_i`) and
    `ν² = Σ π_i²` (`< 1` required), the LRR coefficients are
    `R = (1/(1−ν²)) Σ_i π_i U_i^∇`; the series is extended recursively. **[verify form vs paper]**
  - **Vector (V-) forecast:** continue the trajectory *vectors* inside the signal column space,
    then diagonal-average — typically more stable for multi-step horizons; the 2020 paper
    proposes an improved variant. **[verify the new algorithm's specifics]**
  These define `forecast.py` (Day 41) and the out-of-sample protocol (Phase 4).

- **Long series / scalability** — *Rodrigues, Tuy & Mahmoudvand (2018), "Randomized SSA for
  long time series", J. Stat. Comp. Sim.* Randomized SVD range-finding accelerates the
  decomposition; relevant to the Day-22 performance pass and to robust variants whose cost is
  dominated by the SVD.

- **Missing data** — *Rodrigues & de Carvalho (2013), "Spectral modeling of time series with
  missing data", Appl. Math. Modelling.* Spectral/SSA imputation; fallback for gaps when
  aligning real panels (Phase 3, Days 27–28).

---

## 5. Application context

- **Macro / business cycle** — *de Carvalho, Rodrigues & Rua (2012), "Tracking the US business
  cycle with SSA", Economics Letters.* Motivates the **FRED macro panel** (industrial
  production, CPI, unemployment, yield spreads) and the interpretation of extracted factors as
  cycle phases (Day 30).
- **Equity panel** — daily returns for 15–20 global indices (S&P 500, FTSE 100, Nikkei 225,
  DAX, …), 2005–2024, spanning the GFC, 2015–16 selloff, and COVID — i.e. **outlier-rich by
  construction**, the setting where robustness should matter most.

---

## 6. Methodological positioning

| Method | Multivariate? | Robust? | Reference |
|--------|:---:|:---:|-----------|
| Univariate SSA | ✗ | ✗ | Golyandina et al. (2001) |
| MSSA | ✓ | ✗ | Rodrigues & Mahmoudvand (2018) |
| Robust SSA (univariate) | ✗ | ✓ | Kazemi & Rodrigues (2023) |
| **Robust MSSA (this project)** | ✓ | ✓ | **proposed** |

The contribution is the **bottom-right cell** and its empirical validation on contaminated
financial panels.

---

## 7. Open methodological questions for the group (Day 10 checkpoint)
1. Which robust SVD variant should be primary (ideally the one in Kazemi & Rodrigues 2023 for
   direct comparability)? **[verify]**
2. Are trajectory-matrix outliers best modelled as column-structured (date-driven) or cell-wise?
3. Is a Hankel-structure-preserving robust low-rank step worth the extra cost, or is
   unstructured robust SVD + diagonal averaging adequate?
4. Recurrent vs vector forecast as the primary OOS protocol for the multivariate setting?

---

## References
1. Golyandina, Nekrutkin & Zhigljavsky (2001). *Analysis of Time Series Structure: SSA and Related Techniques.* Chapman & Hall/CRC.
2. Rodrigues, P.C. & Mahmoudvand, R. (2018). *The benefits of MSSA over the univariate version.* Journal of the Franklin Institute.
3. Kazemi, M. & Rodrigues, P.C. (2023). *Robust SSA: comparison between classic and robust approaches for model fit and forecasting.* Computational Statistics.
4. Neto, E.A.L. & Rodrigues, P.C. (2022). *Kernel robust singular value decomposition.* Expert Systems with Applications.
5. Rodrigues, P.C. & Mahmoudvand, R. (2020). *A new approach for the vector forecast algorithm in SSA.* Communications in Statistics: Simulation and Computation.
6. Rodrigues, P.C., Tuy, P.G.S.E. & Mahmoudvand, R. (2018). *Randomized SSA for long time series.* Journal of Statistical Computation and Simulation.
7. Rodrigues, P.C. & de Carvalho, M. (2013). *Spectral modeling of time series with missing data.* Applied Mathematical Modelling.
8. de Carvalho, M., Rodrigues, P.C. & Rua, A. (2012). *Tracking the US business cycle with SSA.* Economics Letters.
9. Candès, Li, Ma & Wright (2011). *Robust Principal Component Analysis?* Journal of the ACM.
10. Zhang, Shen & Huang (2013). *Robust regularized SVD (RobRSVD).* (robust + smoothed singular vectors).
