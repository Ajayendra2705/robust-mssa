# Day-wise Implementation Plan
## Robust Multivariate Singular Spectrum Analysis (Robust MSSA) for Latent Factor Extraction in Cross-Sectional Financial Time Series

**Intern:** Ajayendra Kumar Bansod (IIT Kharagpur)
**Supervisor:** Prof. Paulo Canas Rodrigues (UFBA / SaLLy) + assigned PhD collaborator
**Window:** 12 weeks, 1 Jun 2026 → ~23 Aug 2026 · IST, fully remote
**Cadence assumed:** 5 working days/week → **60 working days**. Weeks map to the proposal's timeline table.

> **Note on current status (20 Jun 2026):** Calendar week 3 is in progress, but the repo holds only proposal docs and no code. This plan is written from **Day 1** so nothing is skipped; if Weeks 1–2 are partly done, compress them and pull the schedule forward. Day numbers are *working* days, not calendar days.

---

## Guiding principles
- **Modularity first:** the SVD step is an interchangeable component (standard SVD ↔ robust SVD ↔ kernel robust SVD). Everything downstream must not care which is plugged in. This is the core architectural commitment from the proposal.
- **Reproduce before you innovate:** a trustworthy standard-MSSA baseline must exist and be validated before Robust MSSA is built.
- **Synthetic ground truth before real data:** validate correctness where the true factor structure is known, *then* move to equity/macro panels.
- **Commit daily, open-source from the start:** GitHub repo public early; every method/experiment reproducible from a script + seed.
- **Defer the estimator choice:** do *not* hard-commit to one robust SVD variant before the methodological discussion with the group (explicit ask in the proposal). Build the slot; fill it after consultation.

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
- Clean `experiments/01_baseline_repro/` into a single reproducible script + config. Write short results note. **Send supervisor/PhD collaborator:** lit note, robust-estimator comparison table, and the explicit question: *which robust SVD variant for the MSSA trajectory-matrix setting?* Tag release `v0.1-baseline`.

---

# PHASE 2 — Robust MSSA & Synthetic Validation (Weeks 3–5 · Days 11–25)
*Proposal deliverable: implement Robust MSSA; validate on synthetic panels with known factor structure and controlled contamination.*

**Day 11 — Synthetic data generator**
- `datasets.py`: generator producing panels `X = S + N + O` with **known** low-rank factor structure `S` (shared trend/cycle/co-movement), idiosyncratic noise `N`, and injectable sparse outliers `O`. Parameterise: #series `p`, #factors `k`, length `T`, noise level, contamination rate ε.

**Day 12 — Ground-truth metrics**
- `metrics.py`: subspace recovery error (principal angles between true vs estimated factor subspaces), reconstruction RMSE vs clean `S`, factor stability. Sanity-check that standard MSSA achieves ~0 subspace error at ε=0.

**Day 13 — Robust SVD backend #1**
- Implement the first robust SVD variant agreed with the group (or a sensible default, e.g. robust PCA / alternating L1 / IRLS-weighted SVD) behind the **same `decompose` interface**. No changes to `mssa.py` should be required — this validates the modularity claim.

**Day 14 — Robust SVD backend #1: correctness**
- Unit-test the robust backend: on uncontaminated data it should ≈ standard SVD; on a single planted outlier it should down-weight rather than rotate the leading subspace. Document convergence behaviour and cost.

**Day 15 — Column-wise Robust SSA (the second baseline)**
- Implement univariate Robust SSA applied series-by-series (the proposal's middle comparator: robust but discards multivariate structure). This becomes baseline #2 against standard MSSA and Robust MSSA.

**Day 16 — First three-way comparison on synthetic (low contamination)**
- Run standard MSSA vs column-wise Robust SSA vs Robust MSSA at ε∈{0, 1%}. Confirm Robust MSSA ≈ standard MSSA when clean, and begins to separate as ε rises.

**Day 17 — Contamination sweep ε ∈ {1%, 5%, 10%, 20%}**
- Full sweep across contamination rates, multiple seeds. Capture subspace error and reconstruction RMSE per method per ε. Store results as tidy CSV + config.

**Day 18 — Sweep over factor structure & dimensions**
- Vary `k` (true #factors), `p` (panel width), `T` (length), and window `L`. Identify regimes where Robust MSSA's gain is largest/smallest. Begin a results table for the report.

**Day 19 — Robust SVD backend #2 (kernel / alternative variant)**
- Add a second robust variant (e.g. kernel robust SVD, Neto & Rodrigues 2022, or a second estimator) — again behind the same interface. This stress-tests modularity and gives a richer comparison.

**Day 20 — Mid-internship checkpoint with supervisor**
- Package synthetic findings (plots: subspace error vs ε per method). Decide with the group which robust variant(s) carry forward to the real-data study. Tag `v0.2-robust-synthetic`.

**Day 21 — Hyperparameter robustness**
- Sensitivity of results to `L`, `r`, and robust-estimator tuning (e.g. weighting threshold). Establish default settings + a small grid for the empirical phase.

**Day 22 — Performance / scaling pass**
- Profile the pipeline; the robust SVD is the bottleneck. Add caching, vectorise embedding/diagonal averaging. Consider randomized SVD init (Rodrigues, Tuy & Mahmoudvand 2018) for long series.

**Day 23 — Test hardening & reproducibility**
- Expand `tests/` (embedding, reconstruction identity, forecast identity, robust-vs-standard equivalence at ε=0). Pin seeds; add a `make repro` / script that regenerates all synthetic results end-to-end.

**Day 24 — Write up synthetic results**
- Draft `report/` section: methodology + synthetic validation with figures and tables. This is the empirical backbone of the eventual paper.

**Day 25 — Phase 2 wrap**
- Code review with PhD collaborator; address feedback. Confirm the three-method harness is frozen and trustworthy before touching real data. Tag `v0.3-validated`.

---

# PHASE 3 — Empirical Study on Real Panels (Weeks 6–8 · Days 26–40)
*Proposal deliverable: empirical study on equity index and macro panels; stability & subspace recovery analysis.*

**Day 26 — Equity panel acquisition**
- Pull daily returns for 15–20 major global indices (S&P 500, FTSE 100, Nikkei 225, DAX, …), 2005–2024 (spans GFC, 2015–16 selloff, COVID). Handle holidays/missing days; align calendars; document data dictionary.

**Day 27 — Equity data QA & preprocessing**
- Returns transformation, missing-data handling (note: spectral imputation à la Rodrigues & de Carvalho 2013 is an option), stationarity checks, outlier *annotation* (don't remove — they're the point). Freeze a processed dataset version.

**Day 28 — Macro panel acquisition (FRED)**
- Monthly US macro series (industrial production, CPI, unemployment, yield spreads) via FRED, following the business-cycle tracking spirit (de Carvalho, Rodrigues & Rua 2012). Align frequencies; document.

**Day 29 — Run all three methods on equity panel**
- Standard MSSA / column-wise Robust SSA / Robust MSSA on the equity panel. Extract leading factors; interpret (market factor, regional/sector co-movements). Plot reconstructed trends through crisis windows.

**Day 30 — Run all three methods on macro panel**
- Same three-method run on the macro panel. Relate extracted factors to known business-cycle phases (recession shading). Qualitative interpretation note.

**Day 31 — Temporal stability: rolling-window design**
- Implement rolling/expanding-window factor extraction. Define a factor-stability metric (subspace overlap of leading factors between adjacent windows; sign/permutation alignment handled).

**Day 32 — Stability analysis on equities**
- Run rolling-window analysis; compare stability of leading factors across the three methods, especially **around crisis episodes** where outliers cluster. Expect Robust MSSA to be more stable — verify.

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
*Proposal deliverable: out-of-sample evaluation; comparison across all three methods and contamination levels.*

**Day 41 — Forecasting backend**
- `forecast.py`: SSA/MSSA recurrent forecast + vector forecast (Rodrigues & Mahmoudvand 2020). Test forecast identities on synthetic signals with known continuation.

**Day 42 — OOS protocol design**
- Define rigorous protocol: train/test split, rolling-origin evaluation, horizons (h=1, 5, 20 for daily; 1,3,6,12 for monthly), metrics (RMSE, MAE, MASE), and significance testing (Diebold–Mariano). Follow the group's prior SSA forecasting methodology.

**Day 43 — OOS run on synthetic**
- Out-of-sample reconstruction/forecast accuracy of the *extracted trend components* vs held-out clean signal, across all three methods × contamination levels. This is the cleanest test of "does robustness translate to predictive content?"

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
- Full, precise methodology section: block-Hankel embedding, the interchangeable SVD step, each robust variant, the three-method comparison design. Notation consistent with the proposal.

**Day 53 — Results write-up (synthetic + empirical)**
- Polish synthetic and empirical sections with final figures, captions, and interpretation.

**Day 54 — Results write-up (OOS) + discussion**
- OOS section + discussion: when/why robustness helps, limitations, threats to validity, future work (e.g. kernel variant, missing-data spectral imputation, hierarchical extension).

**Day 55 — Repository polish for release**
- README with quickstart + figure-reproduction commands, docstrings, `LICENSE` (MIT/BSD), `CITATION.cff`, CI (GitHub Actions running pytest), example notebook. Make sure `pip install -e .` + one script reproduces a headline result.

**Day 56 — Reproducibility audit**
- Fresh-clone test on a clean environment: regenerate key figures from scratch. Fix any hidden state/seed/path issues. Pin all dependency versions.

**Day 57 — Internal review round**
- Send report draft + repo to PhD collaborator/supervisor. Collect feedback.

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
- **Estimator choice stalls progress** → build the modular slot first (Day 13 uses a default); finalise variant after the Day 10/Day 20 discussions.
- **Robust SVD too slow on long series** → randomized SVD init + caching (Day 22); subsample for sweeps.
- **Weak/negative results on real data** → synthetic ground-truth study (Phase 2) still constitutes a publishable methodological contribution; report negatives honestly.
- **Data gaps / non-aligned calendars** → handle explicitly in Days 27–28; spectral imputation available as fallback.
- **Schedule slip (already ~wk3, no code)** → Phase 1 is the compressible buffer; baseline can be accelerated to ~6 days if needed.
