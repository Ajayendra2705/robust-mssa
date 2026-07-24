# Day-wise Implementation Plan
## Robust Multivariate Singular Spectrum Analysis (Robust MSSA) for Latent Factor Extraction in Cross-Sectional Financial Time Series

**Intern:** Ajayendra Kumar Bansod (IIT Kharagpur)
**Supervisor:** Prof. Paulo Canas Rodrigues (UFBA / SaLLy). *No PhD collaborator assigned currently — Prof may have one MSc student join in a few months (confirmed 24 Jul 2026); until then the internship is solo. Steps that assumed a PhD collaborator now route to self-review + supervisor checkpoints.*
**Window:** 12 weeks, 1 Jun 2026 → ~23 Aug 2026 · IST, fully remote
**Cadence assumed:** 5 working days/week → **60 working days**. Weeks map to the proposal's timeline table.

> **Note on current status (24 Jul 2026):** **Phase 1 complete** — modular standard-MSSA baseline implemented, tested, and validated against analytical results and `pyts`; equity + synthetic datasets set up; repo scaffolded (`robust-mssa/`), tagged toward `v0.1-baseline`. Day-10 checkpoint email sent and answered by the supervisor. **Now entering Phase 2 (Robust MSSA).**
>
> **Supervisor guidance received (24 Jul 2026), folded into this plan:**
> 1. **Robust estimator resolved & confirmed.** The attached paper is **Rodrigues, Pimentel, Messala & Kazemi (2020), *Entropy* 22(1):8 — "The Decomposition and Forecasting of Mutual Investment Funds Using SSA"** (`48. 2020 - Entropy - Robust SSA.pdf`). Its **two robust SSA algorithms for model fit** are: **(a) RLSSA — L1-norm robust SVD** (Hawkins, Liu & Young 2001; R `robustSVD()` in *pcaMethods*), and **(b) RHSSA — Huber-function robust SVD**, a special case of robust regularized SVD (Zhang, Shen & Huang 2013; R `RobRSVD(rough=TRUE, uspar=0, vspar=0)`, **d = 1.345**). Both are implemented behind the shared `decompose` interface as `AlternatingL1SVD` (RLSSA) and `RobRSVD` (RHSSA). ✅ Losses/estimators + Huber constant match the paper. ⚠️ **Open validation:** our solver is a joint IRLS-imputation, not the R packages' per-component deflation — cross-check numerical agreement vs `pcaMethods::robustSVD` / `RobRSVD` (Day 14/16) for direct comparability. The paper also gives a **robust forecasting** algorithm (L1 + Huber variants) → Phase 4.
> 2. **Comparison design is now an explicit 2×2 factorial** (see below): **classical vs robust** crossed with **univariate vs multivariate** forecasts.

---

## Guiding principles
- **Modularity first:** the SVD step is an interchangeable component (standard SVD ↔ robust SVD ↔ kernel robust SVD). Everything downstream must not care which is plugged in. This is the core architectural commitment from the proposal.
- **Reproduce before you innovate:** a trustworthy standard-MSSA baseline must exist and be validated before Robust MSSA is built.
- **Synthetic ground truth before real data:** validate correctness where the true factor structure is known, *then* move to equity/macro panels.
- **Commit daily, open-source from the start:** GitHub repo public early; every method/experiment reproducible from a script + seed.
- **Estimator choice (resolved & confirmed 24 Jul 2026):** the modular slot approach paid off — the slot was built in Phase 1; the supervisor's reply fills it with **the two robust SVD algorithms from Rodrigues et al. (2020, *Entropy*)** — L1-norm robust SVD (RLSSA) and Huber robust SVD (RHSSA) — both now implemented behind the unchanged `decompose` interface.

---

## Comparison design — 2×2 factorial (supervisor-directed, 24 Jul 2026)

Every dataset (synthetic + real) and every out-of-sample evaluation runs the full grid, so the two research axes are cleanly separable:

| | **Univariate** (per-series SSA) | **Multivariate** (MSSA, block-Hankel) |
|---|---|---|
| **Classical** (standard SVD) | classical SSA | classical MSSA |
| **Robust** (robust SVD ×2) | robust SSA | **Robust MSSA** ← the project's target method |

- **Axis 1 — classical vs robust:** does replacing the L2 SVD with a robust SVD improve factor recovery / forecasts under contamination?
- **Axis 2 — univariate vs multivariate:** does sharing structure across series (MSSA) beat treating each series alone — and does that gain survive / grow under robustification?
- Both robust SVD algorithms occupy the "robust" row, giving 2 (classical/robust backends) × 2 (uni/multi embedding) with the robust cell instantiated twice → **6 method configurations** per experiment. Report each axis marginally and the interaction.

---

## Repo scaffold (set up Day 1–2, evolve throughout)

```
robust-mssa/
├── README.md
├── pyproject.toml / requirements.txt
├── environment.yml
├── data/
│   ├── raw/            # downloaded equity + FRED series (gitignored)
│   ├── synthetic/      # generated panels with known factor structure
│   └── processed/
├── src/rmssa/
│   ├── __init__.py
│   ├── embedding.py        # trajectory matrix, block Hankel construction
│   ├── decomposition.py    # SVD backends: standard / robust / kernel-robust
│   ├── grouping.py         # eigentriple grouping, w-correlation
│   ├── reconstruction.py   # diagonal averaging / Hankelisation
│   ├── forecast.py         # recurrent + vector forecast
│   ├── mssa.py             # orchestrator: ties stages together
│   ├── metrics.py          # subspace error, w-corr, RMSE/MAE, stability
│   └── datasets.py         # loaders: Yahoo, FRED, synthetic generator
├── experiments/
│   ├── 01_baseline_repro/
│   ├── 02_synthetic_validation/
│   ├── 03_equity_macro_study/
│   ├── 04_oos_evaluation/
│   └── configs/            # yaml per experiment (seed, L, r, contamination)
├── notebooks/          # exploratory only; logic lives in src/
├── tests/              # pytest: embedding, reconstruction, forecast identities
└── report/             # technical report + paper draft (LaTeX)
```

---

# PHASE 1 — Foundations & Baseline (Weeks 1–2 · Days 1–10)
*Proposal deliverable: in-depth review of MSSA & Robust SSA; reproduce standard MSSA baseline on a small financial panel.*

**Day 1 — Setup & kickoff**
- Create public GitHub repo `robust-mssa`, scaffold above, `pyproject.toml`, pinned env (numpy, scipy, pandas, matplotlib, yfinance, pandas-datareader, statsmodels, scikit-learn, pytest).
- Draft README with problem statement (lift from proposal §2). Set up issue tracker + a `PLAN.md` mirroring this file.

**Day 2 — Literature pass 1: SSA / MSSA mechanics**
- Deep read: Rodrigues & Mahmoudvand (2018) *Benefits of MSSA over univariate*; Golub–Reinsch SSA basics. Write a 1-page math note: embedding → SVD → grouping → diagonal averaging, in both univariate and multivariate (block Hankel) form.

**Day 3 — Literature pass 2: Robust SSA / robust SVD**
- Deep read: Kazemi & Rodrigues (2023) *Robust SSA*; skim Neto & Rodrigues (2022) *Kernel robust SVD*. Note each robust estimator's assumptions, breakdown point, and computational cost. Draft a comparison table → this seeds the estimator-choice discussion with the group.

**Day 4 — Literature pass 3: forecasting + context**
- Skim Rodrigues & Mahmoudvand (2020) *vector forecast algorithm*; de Carvalho, Rodrigues & Rua (2012) *business cycle tracking* (motivates the macro panel). Finalise a 2–3 page literature note committed to `report/lit_review.md`.

**Day 5 — Implement embedding + standard SVD backend**
- `embedding.py`: univariate trajectory matrix and multivariate block-Hankel `H ∈ R^{L×K_tot}`. `decomposition.py`: standard SVD backend behind a common interface `decompose(H) -> (U, s, Vt)`. Unit tests for shapes + Hankel structure.

**Day 6 — Implement grouping + reconstruction**
- `grouping.py` (eigentriple grouping, w-correlation matrix), `reconstruction.py` (diagonal averaging / Hankelisation back to series). Test the **reconstruction identity**: full-rank reconstruction recovers the original series to machine precision.

**Day 7 — Wire the standard MSSA orchestrator**
- `mssa.py`: end-to-end `fit → decompose → group → reconstruct`. Config-driven (window `L`, rank `r`, channels). Run on a toy 3-series synthetic signal (trend + seasonal + noise); confirm components separate visually.

**Day 8 — Small real financial panel**
- `datasets.py`: Yahoo loader. Pull a *small* panel (3–5 indices, e.g. S&P 500, FTSE 100, Nikkei 225) for a clean recent window. Apply standard MSSA; extract trend + cyclical components; plot.

**Day 9 — Baseline validation & w-correlation diagnostics**
- Compare against a reference SSA implementation (e.g. `pyts`/R `Rssa` on one series) to confirm correctness. Produce w-correlation heatmaps and scree/eigenvalue plots. Document `L` selection heuristics.

**Day 10 — Phase 1 wrap + checkpoint**
- Clean `experiments/01_baseline_repro/` into a single reproducible script + config. Write short results note. **Sent to supervisor** (lit note, robust-estimator comparison table, and the explicit estimator question). ✅ **Answered 24 Jul 2026:** use the two robust SVD algorithms from his attached paper (Rodrigues et al. 2020, *Entropy* — L1-norm RLSSA + Huber RHSSA); run classical-vs-robust × univariate-vs-multivariate; no PhD collaborator for now. Tag release `v0.1-baseline`.

---

# PHASE 2 — Robust MSSA & Synthetic Validation (Weeks 3–5 · Days 11–25)
*Proposal deliverable: implement Robust MSSA; validate on synthetic panels with known factor structure and controlled contamination.*

**Day 11 — Synthetic data generator**
- `datasets.py`: generator producing panels `X = S + N + O` with **known** low-rank factor structure `S` (shared trend/cycle/co-movement), idiosyncratic noise `N`, and injectable sparse outliers `O`. Parameterise: #series `p`, #factors `k`, length `T`, noise level, contamination rate ε.

**Day 12 — Ground-truth metrics ✅ done**
- `metrics.py`: subspace recovery error (principal angles / `subspace_distance`, `grassmann_distance`, `subspace_overlap`), reconstruction `rmse`/`mae`/`relative_frobenius`, `signal_recovery_error` (vs clean `S`), `factor_stability`. All invariant to sign/rotation/permutation of factors. 13 unit tests (known-value + invariance).

**Day 13 — Robust SVD backend #1 = RHSSA (Huber robust SVD) ✅ done**
- Implemented `RobRSVD` (Huber-function robust SVD, d=1.345; Zhang–Shen–Huang 2013 special case of robust regularized SVD, per Rodrigues et al. 2020) behind the **same `decompose` interface**. No changes to `mssa.py` — modularity claim holds.

**Day 14 — Robust SVD backend #1: correctness ✅ done**
- Unit-tested: clean-data gap vs standard SVD = 0.0 (collapses to L2 at ε=0); planted outlier down-weighted (recovers clean signal better than standard). **Cross-check:** an independent optimiser (robust weighted-ALS, `tests/test_robust_reference.py`) recovers the same robust subspace as the imputation solver on contaminated panels — runnable now, passes. Authoritative package-level check vs R `pcaMethods::robustSVD` / `RobRSVD(rough=TRUE,uspar=0,vspar=0)` is **scripted** (`experiments/02_synthetic_validation/rcheck/`, one command) and runs when R is installed; R is not on this machine, so that run is SKIPPED-pending, not failed.

**Day 15 — Robust SVD backend #2 = RLSSA (L1-norm robust SVD) ✅ done**
- Implemented `AlternatingL1SVD` (L1-norm robust SVD; Hawkins–Liu–Young 2001, R `robustSVD()` in *pcaMethods*, per Rodrigues et al. 2020) behind the same interface. Both robust backends now populate the "robust" row of the 2×2 grid; verified they agree with each other at ε=0 and with the independent ALS reference under contamination. (Note: the earlier plan slotted *kernel* robust SVD here; superseded by the two-algorithm directive — kernel robust SVD, Neto & Rodrigues 2022, demoted to optional extension.)

**Day 16 — First full-grid comparison on synthetic (low contamination) ✅ done**
- `experiments/02_synthetic_validation/run_grid.py` + `configs/grid_synthetic.yaml`: the **2×2 factorial** — {classical, RHSSA-Huber, RLSSA-L1} × {univariate SSA, multivariate MSSA} — at ε∈{0, 1%, 2%}, 5 seeds. Metric = signal-recovery error vs clean `S`. **Results** (`report/results_phase2_grid.md`): ε=0 all six ≈ 0.010 (no robustness tax); classical collapses to 0.35–0.42 at ε=2% while **Robust MSSA (Huber·multi) = 0.013 — best config, ~27× better than classical MSSA**; multivariate beats univariate for every method. All hard checks pass. ⚠️ **Rank lesson:** r must be ≥ signal SSA-rank (here 6) — r=4 undersized and made robust mistake unmodelled signal for outliers (feeds Day 21 rank selection).

**Day 17 — Contamination sweep ε ∈ {1%, 5%, 10%, 20%} ✅ done**
- `experiments/02_synthetic_validation/run_sweep.py` + `configs/sweep_synthetic.yaml`: full 2×2 grid over ε∈{0,1,5,10,20}%, 5 seeds, **two metrics per config from one fit** — reconstruction (`signal_recovery_error`) and **subspace-recovery error** (largest principal angle vs the true clean-signal factor subspace). Tidy CSV + 2-panel plot + JSON. **16/16 checks pass.** **Results** (`report/results_phase2_sweep.md`): classical collapses (recovery >1.0 and subspace ~1.0 by ε≥10%) while **Robust MSSA (Huber·multi) degrades gracefully — recovery 0.016→0.146, subspace 0.011→0.731 over ε=1→20%**; multivariate beats univariate at every ε. Honest limit: robust subspace error reaches 0.73 at ε=20%. Shared experiment logic factored into `_grid_common.py` (reused by Day 16 grid).

**Day 18 — Sweep over factor structure & dimensions ✅ done**
- `experiments/02_synthetic_validation/run_dimsweep.py` + `configs/dimsweep_synthetic.yaml`: OFAT over k∈{1..4}, p∈{3..15}, T∈{150,300,600}, L∈{20..80} at ε=10%, 3 seeds, classical vs Robust MSSA (Huber), rank r=2k+2. **Results** (`report/results_phase2_dimsweep.md`): Robust MSSA wins in **every** regime (gain ≥ 2.9×). **Window L is the strongest lever** (2.9×@L=20 → 68×@L=80); longer T widens the edge (14×→34×); more factors k shrink it (53×@k=1 → 5.7×@k=4); panel width p matters least (~12–33×). Feeds L/rank selection (Day 21). *(Caught+fixed a key-mismatch bug that had frozen L=50 in the L-sweep.)*

**Day 19 — Agreement & diagnostics across the two robust algorithms ✅ done**
- `experiments/02_synthetic_validation/run_algo_compare.py` + `configs/algo_compare.yaml`: Huber vs L1 across ε∈{0,1,5,10,20}%, 5 seeds. Added `n_iter_`/`converged_` diagnostics to `RobustSVD` (unit-tested). **Results** (`report/results_phase2_algocompare.md`): the two are **near-equivalent in accuracy** (inter-algorithm subspace divergence < 0.006 even at ε=20%; recovery within ~5%); agree at ε=0 (divergence 0.0009); divergence grows monotonically with ε. **Huber is cheaper** (fewer iters — 34 vs 57 clean — and faster). Convergence caveat: both approach the 300-iter cap at ε≥10%. **Recommendation: default to Huber (RHSSA)**, keep L1 as the reported second algorithm. 3/3 checks pass. (Kernel robust SVD stretch not needed.)

**Day 20 — Mid-internship checkpoint with supervisor ✅ done**
- Consolidated synthetic findings (Days 11–19) into `report/phase2_summary.md` (the checkpoint package) + drafted the supervisor email `report/checkpoint_email_day20.md`. Confirmed direction: both robust algorithms carry forward, **Huber primary / L1 secondary**; open item flagged for the group — run the R-package cross-check or accept the IRLS-imputation solver substitution. Committed Phase 2 and tagged **`v0.2-robust-synthetic`**. 99 tests pass, ruff clean.

**Day 21 — Hyperparameter robustness**
- Sensitivity of results to `L`, `r`, and robust-estimator tuning (e.g. weighting threshold). Establish default settings + a small grid for the empirical phase.

**Day 22 — Performance / scaling pass**
- Profile the pipeline; the robust SVD is the bottleneck. Add caching, vectorise embedding/diagonal averaging. Consider randomized SVD init (Rodrigues, Tuy & Mahmoudvand 2018) for long series.

**Day 23 — Test hardening & reproducibility**
- Expand `tests/` (embedding, reconstruction identity, forecast identity, robust-vs-standard equivalence at ε=0). Pin seeds; add a `make repro` / script that regenerates all synthetic results end-to-end.

**Day 24 — Write up synthetic results**
- Draft `report/` section: methodology + synthetic validation with figures and tables. This is the empirical backbone of the eventual paper.

**Day 25 — Phase 2 wrap**
- Self-review + supervisor checkpoint (no PhD collaborator assigned yet); address feedback. Confirm the **six-config 2×2 harness** is frozen and trustworthy before touching real data. Tag `v0.3-validated`.

---

# PHASE 3 — Empirical Study on Real Panels (Weeks 6–8 · Days 26–40)
*Proposal deliverable: empirical study on equity index and macro panels; stability & subspace recovery analysis.*

**Day 26 — Equity panel acquisition**
- Pull daily returns for 15–20 major global indices (S&P 500, FTSE 100, Nikkei 225, DAX, …), 2005–2024 (spans GFC, 2015–16 selloff, COVID). Handle holidays/missing days; align calendars; document data dictionary.

**Day 27 — Equity data QA & preprocessing**
- Returns transformation, missing-data handling (note: spectral imputation à la Rodrigues & de Carvalho 2013 is an option), stationarity checks, outlier *annotation* (don't remove — they're the point). Freeze a processed dataset version.

**Day 28 — Macro panel acquisition (FRED)**
- Monthly US macro series (industrial production, CPI, unemployment, yield spreads) via FRED, following the business-cycle tracking spirit (de Carvalho, Rodrigues & Rua 2012). Align frequencies; document.

**Day 29 — Run the full method grid on the equity panel**
- Standard MSSA / column-wise Robust SSA / Robust MSSA on the equity panel. Extract leading factors; interpret (market factor, regional/sector co-movements). Plot reconstructed trends through crisis windows.

**Day 30 — Run the full method grid on the macro panel**
- Same 2×2 grid (classical/robust×2 × uni/multi) on the macro panel. Relate extracted factors to known business-cycle phases (recession shading). Qualitative interpretation note.

**Day 31 — Temporal stability: rolling-window design**
- Implement rolling/expanding-window factor extraction. Define a factor-stability metric (subspace overlap of leading factors between adjacent windows; sign/permutation alignment handled).

**Day 32 — Stability analysis on equities**
- Run rolling-window analysis; compare stability of leading factors across the method grid, especially **around crisis episodes** where outliers cluster. Expect Robust MSSA to be more stable — verify.

**Day 33 — Stability analysis on macro**
- Same on macro panel. Tabulate stability metrics per method. Flag any regimes where robustness does *not* help (honest reporting).

**Day 34 — Subspace recovery proxy on real data**
- With no ground truth, use proxies: stability under sub-sampling/bootstrap, sensitivity to held-out series, agreement across robust variants. Quantify how much standard MSSA's leading subspace is perturbed by identified outlier dates vs Robust MSSA.

**Day 35 — Controlled contamination on real data**
- Inject synthetic outliers into the *real* panels at ε∈{1%,5%,10%,20%} on top of natural contamination, to measure marginal robustness gain in a realistic setting.

**Day 36 — Results consolidation**
- Aggregate all Phase 3 metrics into master result tables (per dataset × method × metric). Generate publication-quality figures (factor trajectories, stability curves, contamination-response curves).

**Day 37 — Interpretation & narrative**
- Write the empirical-findings narrative: where and why Robust MSSA wins, magnitude of gains, failure modes. Connect back to the proposal's research question.

**Day 38 — Robustness/ablation checks**
- Ablations: window length, #factors retained, robust-estimator variant, return vs price space. Confirm conclusions aren't artifacts of one configuration.

**Day 39 — Checkpoint with supervisor**
- Present empirical results; collect feedback on framing and any additional comparisons the group wants. Tag `v0.4-empirical`.

**Day 40 — Phase 3 wrap**
- Freeze experiment configs for `03_equity_macro_study/`; ensure every figure regenerates from script + config + seed.

---

# PHASE 4 — Out-of-Sample Evaluation (Weeks 9–10 · Days 41–50)
*Proposal deliverable: out-of-sample evaluation; comparison across the full 2×2 method grid (classical/robust × uni/multi) and contamination levels.*

**Day 41 — Forecasting backend**
- `forecast.py`: SSA/MSSA recurrent forecast + vector forecast (Rodrigues & Mahmoudvand 2020). Test forecast identities on synthetic signals with known continuation.

**Day 42 — OOS protocol design**
- Define rigorous protocol: train/test split, rolling-origin evaluation, horizons (h=1, 5, 20 for daily; 1,3,6,12 for monthly), metrics (RMSE, MAE, MASE), and significance testing (Diebold–Mariano). Follow the group's prior SSA forecasting methodology.

**Day 43 — OOS run on synthetic**
- Out-of-sample reconstruction/forecast accuracy of the *extracted trend components* vs held-out clean signal, across the full method grid × contamination levels. This is the cleanest test of "does robustness translate to predictive content?"

**Day 44 — OOS run on equity panel**
- Rolling-origin OOS evaluation on equities; per-method, per-horizon error tables. Pay attention to performance in crisis vs calm sub-periods.

**Day 45 — OOS run on macro panel**
- Same on macro panel. Aggregate into the master OOS results table.

**Day 46 — Statistical comparison**
- Diebold–Mariano (or model-confidence-set) tests across methods. Establish where Robust MSSA's improvements are statistically significant vs incidental.

**Day 47 — Contamination × OOS interaction**
- Cross-tabulate OOS accuracy against contamination level: does the robustness advantage grow with ε as hypothesised? Produce the headline figure of the project.

**Day 48 — Sensitivity & honest negative results**
- Document configurations where Robust MSSA does not beat standard MSSA (e.g. very low contamination). Strengthens credibility.

**Day 49 — Checkpoint with supervisor**
- Present full OOS story end-to-end. Decide jointly whether results warrant a journal draft (proposal targets *Computational Statistics* / *JSCS*). Tag `v0.5-oos`.

**Day 50 — Phase 4 wrap**
- Finalise all result artifacts; freeze `04_oos_evaluation/`. Master results spreadsheet complete (synthetic + real + OOS).

---

# PHASE 5 — Report, Release & Paper Draft (Weeks 11–12 · Days 51–60)
*Proposal deliverable: technical report; open-source code release; journal paper draft if results warrant.*

**Day 51 — Report skeleton**
- Assemble `report/` (LaTeX): Abstract, Intro/Motivation (proposal §1–2), Methodology (MSSA + robust SVD), Synthetic validation, Empirical study, OOS evaluation, Discussion, Conclusion. Slot in figures/tables already produced.

**Day 52 — Methods write-up**
- Full, precise methodology section: block-Hankel embedding, the interchangeable SVD step, both robust SVD algorithms, the 2×2 (classical/robust × uni/multi) comparison design. Notation consistent with the proposal.

**Day 53 — Results write-up (synthetic + empirical)**
- Polish synthetic and empirical sections with final figures, captions, and interpretation.

**Day 54 — Results write-up (OOS) + discussion**
- OOS section + discussion: when/why robustness helps, limitations, threats to validity, future work (e.g. kernel variant, missing-data spectral imputation, hierarchical extension).

**Day 55 — Repository polish for release**
- README with quickstart + figure-reproduction commands, docstrings, `LICENSE` (MIT/BSD), `CITATION.cff`, CI (GitHub Actions running pytest), example notebook. Make sure `pip install -e .` + one script reproduces a headline result.

**Day 56 — Reproducibility audit**
- Fresh-clone test on a clean environment: regenerate key figures from scratch. Fix any hidden state/seed/path issues. Pin all dependency versions.

**Day 57 — Internal review round**
- Send report draft + repo to the supervisor (and MSc student if one has joined by then). Collect feedback.

**Day 58 — Revise on feedback**
- Incorporate review comments into report and code. Tighten figures and claims to what the data supports.

**Day 59 — Paper draft assembly (if warranted)**
- If results are strong, format the report into a journal draft for *Computational Statistics* / *JSCS*: trim, sharpen contribution statement, complete references, author/affiliation details.

**Day 60 — Final delivery & handoff**
- Tag `v1.0`, publish release, finalise technical report PDF. Write a handoff/next-steps note. Request the internship confirmation letter for university records (per email thread). Send final package to supervisor.

---

## Milestones / tags at a glance
| Tag | Day | Meaning |
|-----|-----|---------|
| v0.1-baseline | 10 | Standard MSSA reproduced & validated |
| v0.2-robust-synthetic | 20 | Robust MSSA working on synthetic |
| v0.3-validated | 25 | Three-method harness frozen |
| v0.4-empirical | 39 | Real-panel study complete |
| v0.5-oos | 49 | Out-of-sample evaluation complete |
| v1.0 | 60 | Report + release + paper draft |

## Key risks & mitigations
- **Estimator choice** → ✅ resolved & confirmed (24 Jul 2026): the two robust SVD algorithms of Rodrigues et al. (2020, *Entropy*) — L1-norm (RLSSA) and Huber (RHSSA) — both implemented. Residual risk: our IRLS-imputation solver differs from the R packages; validate numerical agreement vs `pcaMethods::robustSVD` / `RobRSVD` (Day 14/16).
- **No PhD collaborator assigned** → internship is solo for now (possible MSc student in a few months). Code-review/checkpoint steps route to self-review + supervisor; nothing blocks on a collaborator.
- **Robust SVD too slow on long series** → randomized SVD init + caching (Day 22); subsample for sweeps.
- **Weak/negative results on real data** → synthetic ground-truth study (Phase 2) still constitutes a publishable methodological contribution; report negatives honestly.
- **Data gaps / non-aligned calendars** → handle explicitly in Days 27–28; spectral imputation available as fallback.
- **Schedule slip (already ~wk3, no code)** → Phase 1 is the compressible buffer; baseline can be accelerated to ~6 days if needed.
