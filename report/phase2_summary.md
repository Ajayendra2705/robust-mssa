# Phase 2 — Robust MSSA & Synthetic Validation (Days 11–19)

**Mid-internship checkpoint · tag `v0.2-robust-synthetic`**
Intern: Ajayendra Kumar Bansod (IIT Kharagpur) · Supervisor: Prof. Paulo Canas Rodrigues (UFBA)

---

## 1. What was built

Following the supervisor's 24 Jul 2026 directive, the interchangeable SVD slot from
Phase 1 is now filled with the **two robust SVD algorithms of Rodrigues, Pimentel,
Messala & Kazemi (2020, *Entropy* 22(1):8)**, both behind the unchanged `decompose`
contract (the modularity claim holds — `mssa.py` was not touched):

| Backend | Paper method | Robust SVD | Reference |
|---|---|---|---|
| `RobRSVD` | **RHSSA** | Huber-function robust SVD, d = 1.345 | Zhang, Shen & Huang (2013); R `RobRSVD` |
| `AlternatingL1SVD` | **RLSSA** | L1-norm robust SVD | Hawkins, Liu & Young (2001); R `pcaMethods::robustSVD` |

Both share one engine — **IRLS by reweighted imputation**: weight residuals against the
full rank-r model, pull down-weighted cells toward the model, re-truncate by SVD. This
collapses to the ordinary SVD when there is nothing to down-weight, so classical-vs-robust
is a fair comparison. Supporting infrastructure: synthetic panel generator (known
`X = S + N + O`), a metrics module (subspace recovery via principal angles, reconstruction
RMSE/relative-Frobenius, factor stability), and `n_iter_`/`converged_` diagnostics.

## 2. Comparison design (supervisor-directed)

The **2×2 factorial** — {classical, robust×2} × {univariate SSA, multivariate MSSA} —
is evaluated on synthetic panels with a **known** clean low-rank signal `S`, scored by two
ground-truth metrics: signal-recovery error and subspace-recovery error.

## 3. Headline results

**(Day 16) No robustness tax at ε=0; robust wins under contamination.** All six configs
recover the clean signal to ~0.01 at ε=0; at ε=2% classical MSSA degrades to 0.35 while
**Robust MSSA (Huber·multivariate) = 0.013 (~27× better)**. `report/results_phase2_grid.md`.

**(Day 17) Contamination sweep ε ∈ {0,1,5,10,20}%.** Classical collapses fast (recovery
> 1.0 and subspace angle ≈ 1.0 by ε≥10%); Robust MSSA degrades gracefully (recovery
0.016→0.146, subspace 0.011→0.731 over ε=1→20%). Multivariate beats univariate at every
level. `report/results_phase2_sweep.md`.

**(Day 18) Where the gain is largest/smallest** (gain = classical/robust recovery error,
ε=10%). Robust MSSA wins in **every** regime (gain ≥ 2.9×):
- **Window L** is the strongest lever: 2.9× (L=20) → 68× (L=80).
- **Longer series T** widen the edge: 14× → 34× (T=150→600).
- **More factors k** shrink it: 53× → 5.7× (k=1→4).
- **Panel width p** matters least: ~12–33× across p ∈ {3..15}.

`report/results_phase2_dimsweep.md`.

**(Day 19) Huber vs L1 are near-equivalent; Huber is cheaper.** Their factor subspaces
differ by < 0.006 (largest principal angle) even at ε=20%; recovery within ~5%. Huber
converges in fewer IRLS sweeps and less wall time. **→ Default to Huber (RHSSA)**, keep
L1 (RLSSA) as the reported second algorithm. `report/results_phase2_algocompare.md`.

## 4. Validation & reproducibility

- **99 unit tests pass**, ruff-clean. Robust backends verified to (i) equal the standard
  SVD on clean/exactly-low-rank data, (ii) down-weight planted outliers, (iii) agree with
  an **independent** weighted-ALS solver on contaminated data.
- Every result reproduces from a committed script + config + seed under
  `experiments/02_synthetic_validation/` and `experiments/configs/`.

## 5. Open items / decisions for the checkpoint

1. **R package cross-check pending.** The estimators and the Huber constant match the
   paper; our solver is a joint IRLS-imputation rather than the R packages' per-component
   deflation. A one-command cross-check vs `pcaMethods::robustSVD` / `RobRSVD` is scripted
   (`experiments/02_synthetic_validation/rcheck/`) but **R is not installed locally** —
   would value the group running it or confirming the solver substitution is acceptable.
2. **Both algorithms carry forward** to Phase 3 (per directive), with **Huber as primary**
   given the Day-19 cost/accuracy result — confirm this framing.
3. **Rank / window selection.** Results are sensitive to r (must be ≥ signal SSA-rank) and
   L (biggest lever). Day 21 will formalise selection heuristics.
4. **Convergence under heavy contamination.** At ε ≥ 10% the IRLS solvers approach the
   iteration cap at tol=1e-8; a looser tol reaches the same accuracy far sooner. Fine for
   the study; noting it for the write-up.

## 6. Next (Phase 3, Days 26–40)

Empirical study on the equity index panel (already downloaded, 2005–2024) and a FRED
macro panel: run the full 2×2 grid, extract and interpret leading factors, rolling-window
stability around crisis episodes, and controlled contamination on real data.
