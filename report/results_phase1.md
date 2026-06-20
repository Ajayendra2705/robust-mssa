# Phase 1 results note (Days 1–10)

**Deliverable:** in-depth review of MSSA & Robust SSA; a working, tested **standard MSSA
baseline**; reproducible experiment scaffold. Tag: `v0.1-baseline`.

---

## 1. What exists now

A complete, tested standard (M)SSA pipeline behind a pluggable decomposition backend —
the architecture the robust variant will slot into without touching downstream code.

| Module | Stage | Status |
|--------|-------|--------|
| `embedding.py` | trajectory / block-Hankel matrix | done, tested |
| `decomposition.py` | SVD backend (`StandardSVD`) + `DecompositionBackend` interface | done, tested |
| `grouping.py` | w-correlation, elementary components, auto-grouping | done, tested |
| `reconstruction.py` | diagonal averaging (uni + per-channel MSSA) | done, tested |
| `mssa.py` | `MSSA` orchestrator (uni + multivariate, config-driven) | done, tested |
| `datasets.py` | synthetic panel generator + cached Yahoo loader | done (synthetic tested) |
| `plots.py` | scree / w-correlation / component figures | done |

**Tests: 52 passing** (`pytest`). Key correctness guarantees asserted:
- Hankel + block-Hankel structure of the embedding.
- SVD fidelity `Σ_i s_i U_i V_iᵀ = H`; rank-`r` truncation error = `σ_{r+1}` (Eckart–Young).
- **Reconstruction identity:** full-grouping diagonal averaging returns the original series
  (univariate and per-channel MSSA) to ~1e-9 or better.
- w-correlation: unit diagonal, symmetric, trend/oscillation separable, sinusoid pairs coupled.
- Synthetic panel: clean signal is exactly rank `k`; contamination rate matches request.

## 2. Baseline reproduction

`experiments/01_baseline_repro/run_baseline.py` + `experiments/configs/baseline.yaml`.

- **Intended data:** daily returns for `^GSPC, ^FTSE, ^N225, ^GDAXI`, 2005–2024 (Yahoo).
- **This run:** Yahoo was rate-limited/unreachable in the build environment, so the script's
  **offline synthetic fallback** was used (T=2000, p=4, k=2, clean). This is logged in
  `outputs/summary.json` (`"source": "synthetic"`). Re-run with network/cache to use the real
  panel — the loader caches to `data/raw/` on first success.
- **Result (synthetic fallback, L=250):** numerical rank 250; variance contributions
  61.16% / 17.21% / 17.18% / 1.26% / … — i.e. a dominant trend factor plus the expected
  sinusoid **pair** (~17%+17%), matching the planted k=2 structure. Reconstruction relative
  error **6.5e-16**.
- Figures written: `scree.png`, `wcorrelation.png`, `components_channel0.png`.

> Validation against an external reference (R `Rssa` / `pyts`) is **pending** — deferred from
> Day 9 because it needs the real downloaded panel; will be completed when Yahoo access is
> available (or from cache) early in Phase 2. The internal reconstruction-identity and
> Eckart–Young tests already pin correctness of the core algebra.

## 3. Reading / methodology notes produced
- `ssa_math_notes.md` — four-stage (M)SSA derivation + implementation contract.
- `robust_svd_comparison.md` — 7 robust-SVD candidates, keyed to the column-structured
  outlier geometry of the trajectory matrix; recommendation + open questions.
- `lit_review.md` — consolidated review positioning Robust MSSA in the (multivariate × robust)
  grid, with forecasting / scalability / missing-data / business-cycle context.
- `L_selection.md` — window-length heuristics and the Phase-2 sweep plan.

## 4. Open questions for the supervisor / PhD collaborator (checkpoint)
1. **Robust SVD variant** — which does Kazemi & Rodrigues (2023) use; should that be our
   primary estimator for direct comparability? (see `robust_svd_comparison.md`)
2. **Outlier geometry** — column-structured (date-driven) vs cell-wise? Determines whether
   PCP-style sparsity assumptions apply.
3. **Hankel-preserving robust step** — worth the cost, or is unstructured robust SVD +
   diagonal averaging adequate?
4. **MSSA stacking** — confirm horizontal (common-window) form is the intended baseline
   (matches the proposal's `H ∈ R^{L×K_tot}`); vertical MSSA only as an ablation?
5. **Reference implementation** — any preferred package/version to validate the baseline
   against?

## 5. Carry-over into Phase 2
- Complete the external-reference validation once real data is available.
- Day 11: synthetic generator already supports `contamination` + ground-truth `mask`/`signal`
  → ready for subspace-recovery metrics (Day 12) and the first robust backend (Day 13).
- The `DecompositionBackend` interface is the only thing the robust work needs to implement.
