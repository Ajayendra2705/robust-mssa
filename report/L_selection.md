# Window-length (L) selection — heuristics note (Day 9)

The window length `L` is the one structural hyperparameter of (M)SSA. It sets the
trade-off between **frequency resolution** (large `L`) and **statistical stability** of the
estimated subspace (large `K = N − L + 1`, i.e. small `L`). This note records the working
heuristics we use; they are revisited empirically in Phase 2 (Day 21 sensitivity sweep).

## Principles
1. **Bounds.** `1 < L < N`. Practically `L ∈ [N/4, N/2]`. `L = ⌊N/2⌋` maximises resolution
   and the number of recoverable components but is the most expensive and gives the fewest
   columns `K`.
2. **Periodicity alignment.** To separate a periodic component of period `q`, choose `L` to
   be a **multiple of `q`**. For daily financial data, candidate `q`: ~5 (weekly), ~21
   (monthly), ~63 (quarterly), ~250 (annual). The baseline uses `L = 250` (≈ one trading
   year) so the dominant annual cycle is well resolved.
3. **Component count.** With window `L`, at most `L` eigentriples exist; a periodic component
   consumes a **pair**. Pick `L` large enough to host all expected signal components plus
   headroom, but not so large that each component is estimated from too few columns.
4. **Separability check, not a single number.** After fitting, read the **w-correlation
   matrix** (`report/ssa_math_notes.md`): if intended components show large off-diagonal
   |w-corr|, they are mixing — adjust `L` (often increasing it) or regroup.
5. **Stability over a single fit.** Prefer an `L` whose leading subspace is **stable across
   rolling windows** (Phase 3, Day 31). A "resolved but unstable" decomposition is worse than
   a slightly coarser but stable one — especially under contamination, which is the whole
   point of the robust variant.

## MSSA-specific
- All channels share one `L`, so choose it for the **common** structure of interest
  (market/cycle factor), not any single series' idiosyncrasy.
- `K_tot = pK` grows with the number of channels, so the column space is better-determined in
  MSSA than in univariate SSA at the same `L` — one of the reasons MSSA tolerates larger `L`.

## Default decisions (baseline)
| Setting | Value | Rationale |
|---------|-------|-----------|
| Daily equity returns | `L = 250` | resolves the annual cycle; ample components; `K` still large for 2005–2024 (`N ≈ 5000`) |
| Monthly macro (FRED) | `L = 60` (≈5 yr) **[to confirm Phase 3]** | resolves business-cycle horizons; `N ≈ 240–360` |
| Synthetic validation | `L = T/4 … T/2` | swept explicitly on Day 21 |

## To do (Phase 2)
- Day 21: grid `L` over {N/6, N/4, N/3, N/2} × component counts; report subspace-recovery and
  w-correlation separability per `L`; pick defaults that are both resolved and stable, and
  check whether the robust backend shifts the optimal `L` relative to standard SVD.
