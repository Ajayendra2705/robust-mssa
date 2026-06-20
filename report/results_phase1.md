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

**Tests: 64 passing** (62 fast + 2 `external` pyts cross-checks; `pytest` / `pytest -m external`).
Key correctness guarantees asserted:
- Hankel + block-Hankel structure of the embedding.
- SVD fidelity `Σ_i s_i U_i V_iᵀ = H`; rank-`r` truncation error = `σ_{r+1}` (Eckart–Young).
- **Reconstruction identity:** full-grouping diagonal averaging returns the original series
  (univariate and per-channel MSSA) to ~1e-9 or better.
- w-correlation: unit diagonal, symmetric, trend/oscillation separable, sinusoid pairs coupled.
- Synthetic panel: clean signal is exactly rank `k`; contamination rate matches request.
- **Analytic SSA-rank ground truth:** exponential->1, sinusoid->2, linear trend->2, sum of
  two exponentials->2, modulated cosine->2, sum of two sinusoids->4 (all verified).
- **External cross-check vs `pyts`:** full-reconstruction rel. error ~1e-15 and leading
  elementary component correlates 1.000000 with ours (see `validation_reference.md`).

## 2. Baseline reproduction

`experiments/01_baseline_repro/run_baseline.py` + `experiments/configs/baseline.yaml`.

- **Data:** daily log-returns for `^GSPC, ^FTSE, ^N225, ^GDAXI`, 2005–2024 (Yahoo).
- **Yahoo 429 — resolved.** Initial runs failed with HTTP 429 / `YFTzMissingError`; the cause
  was an outdated `yfinance` (0.2.40). Upgrading to `yfinance>=1.4` (plus a `curl_cffi`
  browser-impersonating session in the loader) fixed it. Real panel now downloads: **4619
  trading days × 4 indices**, cached to `data/raw/`. See README "Troubleshooting".
- **Real-data result (L=250):** numerical rank 250; the variance spectrum is **nearly flat**
  (top components ~0.68% each) — i.e. raw daily returns have no dominant low-rank structure,
  as expected for noise-like return series. Reconstruction relative error **2.7e-15**.
  → *Empirical flag for Phase 2/3:* consider analysing prices/levels or a different transform
  where common low-rank co-movement is stronger, and let the contamination study (where the
  robust gain should appear) drive the comparison.
- **Synthetic control (offline fallback, L=250):** with a planted k=2 structure the spectrum
  is sharply low-rank — 61.2% trend + a 17%+17% sinusoid pair — confirming the pipeline
  recovers known structure when it exists.
- Figures written: `scree.png`, `wcorrelation.png`, `components_channel0.png`.

> **External-reference validation (Day 9): done.** Closed via two independent routes that do
> not depend on the real panel — analytic SSA-rank ground truth and a `pyts` cross-check
> (`experiments/01_baseline_repro/validate_reference.py` -> `report/validation_reference.md`,
> all checks pass). The only item still tied to live data is producing the baseline *figures*
> on the real equity panel (below).

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
- Produce the baseline *figures* on the real equity panel once Yahoo is reachable (loader
  caches on first success; the algebra is already reference-validated).
- Day 11: synthetic generator already supports `contamination` + ground-truth `mask`/`signal`
  → ready for subspace-recovery metrics (Day 12) and the first robust backend (Day 13).
- The `DecompositionBackend` interface is the only thing the robust work needs to implement.
