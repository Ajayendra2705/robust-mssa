# Revised simulation design (v2)

Prepared for Prof. Paulo Canas Rodrigues, following his 25 Jul 2026 questions on the
simulation study. This is the "short table before running it at full scale" promised in
my reply of the same date.

Everything below is **implemented and tested** (191 passing tests), and the pilot runs in
§5 have been executed at reduced replication (3–5 seeds per cell, ~2 000 fits). The R
cross-check is also done — §5.4. What remains is the full-scale run; §6 lists the
decisions I would like from you before committing compute to it, because two of them
change what the design should be.

> **Note added 2 Sep.** The R cross-check exposed a fault in my robust solver: its
> iteration is a fixed-point scheme, so it could be captured by the outliers at its
> starting point and never recover. It is fixed (the fit now starts from two points and
> keeps the better one), and clean-data results are unchanged. The §5.1 fit table has been
> re-run under the fix and its conclusion is unchanged; **§5.3 (forecasting) has not been
> re-run yet** and its numbers should be read as provisional.

---

## 1. The design in one table

**What a "seed" varies, and what it does not.** Worth stating explicitly because it
changes how the replication counts should be read: the factor series themselves are
deterministic (fixed sinusoids, or fixed segments of a real series). A seed re-draws the
loadings, the noise and the outlier positions — not the signal. The design is therefore
**paired on the signal**, which is the right thing for comparing methods and contamination
types (the comparison is within-signal and much lower variance), but it means the seed
spread measures contamination and loading variability only. Generalisation across *signal
realisations* is not tested by adding seeds, and would need a second axis. Measured
cross-seed |corr| of the clean panel is 0.27 mean / 0.70 max, i.e. seeds are correlated
replications, not independent ones.

| # | Factor | Levels | Baseline | Why it is here |
|---|--------|--------|----------|----------------|
| 1 | **Base series** (clean signal) | `airpassengers` (144), `co2` (2284), `sunspots` (309), `nile` (100), `synthetic` | `airpassengers` | Your suggestion. The uncontaminated original is the known clean signal, so ground-truth scoring survives the move to real data. `synthetic` (distinct-period sinusoids) is retained where exact rank and exact independence are needed. |
| 2 | **Cross-series dependence** | `independent`, `partial(ρ)`, `shared` | `shared` | The axis the Phase-2 generator lacked entirely. Makes H1/H2 testable. |
| 3 | **Shared variance ratio ρ** | 0.1, 0.5, 0.9 | 0.5 | Dials `partial` continuously between the two extremes. |
| 4 | **Outlier type** | `additive`, `patch`, `level_shift`, `innovational` | `additive` | Your "several types". Standard taxonomy (Fox 1972; Chen & Liu 1993). |
| 5 | **Outlier magnitude** | 3, 5, 8 × sd | 8× | Phase 2 used 8× only, which is generous to the robust estimators. |
| 6 | **Contamination rate ε** | 0, 1, 2, 5, 10, 20 % of cells | — | As before. All four types fill the *same* cell budget, so ε means one thing across types. |
| 7 | **Number of series p** | 3, 4, 6, 10 | 4 | |
| 8 | **Length T** | 144 (AirPassengers), 300, 600 | base-dependent | Capped by the real series' own length. |
| 9 | **Common factors k** | 1, 2, 3 | 2 | |
| 10 | **Window L** | 20, 30, 48, 60, 80 | 48 | Phase 2 found L the strongest single lever. |
| 11 | **Rank r** | 4, 6, 8, 12, 16, 20 | 8 (real), exact (synthetic) | **Promoted to a factor** — see §4, this turned out to matter more than expected. |
| 12 | **Method** | classical, RHSSA (Huber), RLSSA (L1) | — | Huber primary, L1 to the appendix, as you directed. |
| 13 | **Mode** | univariate SSA, multivariate MSSA | — | The second arm of the 2×2. |
| 14 | **Evaluation** | **fit** + **forecast** | both | Answers your "fit or forecasting?" — now both, in one design. |
| 15 | **Seeds** | 5 per cell | | |

**Metrics.** Fit: signal-recovery error ‖Ŝ − S‖_F/‖S‖_F and subspace-recovery error
(largest principal angle vs the true factor subspace). Forecast: rolling origin,
horizons h = 1…12, RMSE and MAE **against the clean signal**, refitting on the
contaminated history at each origin — so no method is rewarded for predicting an outlier.

**Robust forecasting** is not a separate implementation: the recurrent forecast reads its
linear recurrence off whatever subspace the backend produced, so a robust backend yields
a robust LRR applied to a robust reconstruction. Only the backend differs between cells.

---

## 2. Your two hypotheses, as testable statements

* **H1** — robust SSA ≈ robust MSSA when the series are independent.
* **H2** — all four combinations coincide with no contamination *and* independence.

Both are now explicit PASS/FAIL checks in `experiments/03_design_v2/run_h1h2.py`. Two
configurations count as coinciding if they are within 25% of each other *relatively* or
within 0.02 *absolutely*. The absolute clause is needed, not a loophole: at ε = 0 the
errors are ~0.008 and ~0.015 — both essentially exact reconstructions — yet a ratio test
on two near-zero numbers calls that a 63% disagreement. Results in §5.2.

---

## 3. One problem I could not design away: what "independent" means

Constructing *real* series that are independent is harder than it looks, and the three
obvious routes each fail differently. I measured all three rather than assuming:

| Construction | Independent? | Keeps real SSA rank profile? | Measured |
|---|---|---|---|
| Distinct real series (AirPassengers, co2, …) | ✗ | ✓ | trending series co-trend, mean abs corr ≈ 0.65 |
| Non-overlapping windows of one long series | ✗ | ✓ | a periodic series co-cycles with its own later windows, abs corr ≈ 0.83–0.91 between co2 windows |
| Phase-randomised surrogates (Theiler et al. 1992) | ✓ by construction | ✗ | AirPassengers reaches 99% of trajectory variance at r=10; **its surrogate needs r=28** |

The third row is the one worth flagging. A surrogate preserves the amplitude spectrum —
hence the autocovariance — but AirPassengers' *low SSA rank* comes from deterministic
trend-plus-seasonal structure, and a random-phase realisation with the same power
spectrum spreads that energy over far more singular triples. So surrogates buy
independence at the cost of the low-rank structure that makes this an SSA problem.

There is also a subtler point. Even with genuinely independent generating processes, the
realised *sample* correlation stays high at moderate T for an autocorrelated base —
mean |corr| ≈ 0.35 for AirPassengers-based surrogates at T = 144 — the Yule (1926)
nonsense-correlation effect, since heavy low-frequency power shrinks the effective
sample size. The signed correlation does average to ≈ 0, as it must.

**How I have resolved it for now:** the dependence axis (H1/H2) is carried by the
*synthetic* base, where independence is exact (mean |corr| ≈ 0.03) and rank is exact;
the *real* base runs the same grid as a sensitivity check with the residual correlation
reported in every table. Both statistics (`mean_abs_corr`, `max_abs_corr`) are attached
to every generated panel. This is question (a) in §6.

---

## 4. Rank selection turned out to be a first-class factor

Phase 2 established a lower bound: r below the signal's SSA-rank makes the robust fit
mistake unmodelled signal for outliers. The real-series work has now established an
upper bound, and it bites harder:

| r | clean-data floor | classical MSSA | Robust MSSA (Huber) | raw gain | **net of floor** |
|---|---|---|---|---|---|
| 4 | 0.2632 | 0.6048 | 0.2718 | 2.2× | 8.1× |
| 6 | 0.1851 | 0.7020 | 0.1914 | 3.7× | **13.9×** |
| 8 | 0.1437 | 0.8008 | 0.1824 | 4.4× | 7.0× |
| 12 | 0.1128 | 0.9752 | 0.3974 | 2.5× | 2.5× |
| 16 | 0.0893 | 1.1236 | 0.5714 | 2.0× | 2.0× |
| 20 | 0.0717 | 1.2564 | 0.7467 | 1.7× | 1.7× |

_(AirPassengers + co2 panel, L = 48, additive outliers, ε = 5%, magnitude 8×, 3 seeds.)_

A rank-r model in an L-dimensional space with r approaching L can fit the outliers
*exactly*, so the robust advantage decays. The effect is not a floor artefact — it is
*stronger* net of the floor, since the floor falls with r while the robust advantage falls
faster. Two consequences:

* **The optimum is near r = 6, and the main grid's r = 8 is already slightly past it.**
  Worth re-centring the rank grid before the full-scale run.
* **An automatic "keep 99.9% of variance" rule is unusable on real data.** It selects
  r = 45 out of a possible 48 for AirPassengers, at which point classical and robust are
  equally bad (recovery error ≈ 1.8) and the comparison measures nothing at all.

Hence factor 11. Note that the *structured* types invert the pattern: for patches the
robust gain is largest at the most restrictive rank (2.2× at r = 4) and flat at ~1.1×
everywhere above it, so there is no rank at which robustness rescues a patch.

---

## 5. Pilot results

Full tables in `report/results_v2_h1h2.md`, `results_v2_contamination.md`,
`results_v2_forecast.md`, `results_rcheck.md`.

### 5.1 The headline: the Phase-2 result is specific to *isolated additive* outliers

Signal-recovery error, AirPassengers+co2 panel, classical MSSA vs Robust MSSA (Huber),
L = 48, r = 8:

Two ratios are given per cell. **raw** is classical error ÷ robust error. **net** removes
the approximation floor first — with a real base series at r = 8 no method can do better
than 0.1437 on clean data, so a raw ratio understates a robust fit that is already sitting
on the floor. Writing `excess = sqrt(error² − floor²)`, **net** = excess_classical ÷ excess_robust.

Both are needed because raw ratios are not comparable across setups. Phase 2's headline
27× (0.347 ÷ 0.013 at ε = 2%) was itself a raw ratio against a floor of 0.010; net of that
floor it is ≈ 42×. The panels also differ — Phase 2 used a synthetic base, this uses a real
one — so the cross-reference below is context, not a controlled comparison. The *type*
comparison within this table is controlled: same panel, same r, same floor throughout.

Re-run 2 Sep under the corrected solver (§5.4). The earlier figures are kept in the right
column so the effect of the fix is visible:

| outlier type (8× sd) | ε = 1% | ε = 5% | ε = 10% | ε = 20% | _(before the fix)_ |
|---|---|---|---|---|---|
| **additive** | 2.5× raw / **7.7× net** | 5.2× / 16.7× | 4.2× / 5.2× | 2.2× / 2.2× | _9.1 / 7.0 / 2.5 / 1.9 net_ |
| **innovational** | 1.6× / 1.8× | 1.5× / 1.6× | 1.5× / 1.5× | 1.4× / 1.4× | _1.9 / 1.5 / 1.4 / 1.4_ |
| **patch** | 1.7× / 2.0× | 1.1× / 1.1× | 1.0× / 1.0× | 1.1× / 1.1× | _1.1 / 1.1 / 1.0 / 1.0_ |
| **level shift** † | 1.0× / 1.0× | 1.0× / 1.0× | 1.0× / 1.0× | 1.0× / 1.0× | _unchanged_ |

† For level shifts the ε column is nominal — see the parametrisation note below. Realised
rates are 15%, 15%, 21.9%, 25.4%.

**The ordering is unchanged by the fix**: additive is separated from the structured types
at every rate, and level shifts sit at 1.00–1.01× throughout. What the fix moved is the
additive row (larger, and no longer collapsing at ε = 10%) and the seed scatter.

**How much of this is seed noise?** This was the weakest part of the pilot, and the fix
substantially answers it. The per-seed additive gains that motivated the caveat —
25.6, 16.4, 4.0 at ε = 5% — turned out to be mostly the solver being captured on the low
seeds, not sampling variability. On the Phase-2 grid, where the same effect is measurable
over 10 seeds, the max/min spread falls from 8.1× to 2.2× at ε = 5% and from 5.5× to 1.2×
at ε = 2%. The structured types were near-noiseless before and remain so (patch
1.06/1.09/1.05, level shift 1.00/1.01/1.00).

Two things still owed: seeds here are paired on the signal, so a second *signal
realisation* remains the real generalisation check (§5.1 does this across four base
signals, with identical ordering), and the additive multiplier still deserves many more
seeds before any single number is quoted.

**Does it generalise beyond this one signal?** Since seeds are paired on the signal, the
check that matters is repeating the comparison on *different* clean signals. Net-of-floor
gain at ε = 5%, magnitude 8×, r = 8:

| clean signal | clean floor | additive | innovational | patch | level shift |
|---|---|---|---|---|---|
| AirPassengers + co2 | 0.1437 | 7.01× | 1.49× | 1.07× | 1.01× |
| co2 + sunspots | 0.3176 | 5.95× | 1.87× | 1.29× | 1.00× |
| sunspots + co2 | 0.2924 | 7.54× | 1.81× | 1.22× | 1.00× |
| synthetic | **0.0103** | 8.10× | 1.50× | 1.19× | 1.00× |

**The ordering is identical in all four**, across signals with wildly different character
and approximation floors spanning 30×. The synthetic row settles the floor question
outright: with a floor of 0.0103 — essentially no approximation error, the Phase-2 regime —
additive still gives 8.1× while level shifts give exactly 1.00×. So the type effect is not
a floor artefact, not a property of AirPassengers, and not a seed accident.

Additive contamination — the only type Phase 2 tested — is the most favourable row here.
Widening the contamination model changes the picture:

* **Patches**: gain collapses to ~1.0, raw and net alike. A run of consecutive
  contaminated points looks locally like signal, and a per-cell M-estimator judging each
  residual against the current fit has no reason to reject it.
* **Level shifts**: exactly 1.0 at every rate and magnitude, with *both* methods failing
  outright (error 1.14–4.01, i.e. worse than predicting zero).

  The mechanism is confirmed directly rather than inferred. At ε = 5% (realised 15%) the
  planted outlier energy is ‖O‖_F/‖S‖_F = **3.027**, and the observed recovery errors are
  **3.006** (Huber) and 3.022 (classical) — the reconstructions retain essentially the
  *entire* shift, rejecting none of it. The contrast with additive contamination is sharp:
  there at ε = 20% the outlier energy is 3.575 while the robust error is 0.84, so that
  energy demonstrably *is* being rejected. A permanent step is not an outlier to a
  per-cell M-estimator; it is 50% contamination within its column, far past breakdown, and
  it gets absorbed into the trend.
* **Innovational**: a modest, stable 1.4–1.9×.
* **Magnitude does _not_ matter much — this corrects a first reading.** On raw ratios the
  additive gain looks like it halves when outliers shrink from 8× to 3× sd (4.4× → 2.1× at
  ε = 5%). Net of the floor that pattern largely dissolves: 7.0× → 6.3× at ε = 5%, and at
  ε = 10–20% the *smaller* outliers actually score slightly higher (3.3× vs 2.5×, 2.4× vs
  1.9×). The apparent magnitude effect was an artefact of a robust fit at 3× sd sitting
  close to the floor, where the raw ratio is compressed. What does move the gain is the
  contamination *rate*: net gain falls from ~9× at ε = 1% to ~2× at ε = 20%, which is
  ordinary breakdown behaviour.

The type effect is the durable finding here, and I would rather report it than the 27×
headline. If it survives the full-scale run, the reading is that the Rodrigues et al.
(2020) robust SVDs are well matched to additive contamination and effectively blind to
structured contamination — a result in itself, and arguably where a methodological
contribution could sit.

**A parametrisation caveat.** For level shifts, ε is a poor control: a single permanent
step in a p = 4 panel already alters ~15% of cells, so target rates of 1% and 5% both
realise 15%. The comparison at equal *realised* contamination still holds (additive at a
realised 20% gives 1.9×; level shift at a realised 25.4% gives 1.0×), so the conclusion is
not an artefact of the mismatch — but the table's ε column cannot be read as
type-comparable for that row. The generator now also accepts an explicit event count
(`n_events`), which is the honest knob for this type. Worth deciding which to report.

### 5.2 H1 and H2

| hypothesis | base | rank convention | A | B | rel. gap | abs. gap | verdict |
|---|---|---|---|---|---|---|---|
| H1 | synthetic | matched | 0.0259 | 0.5605 | 1.823 | 0.5346 | **FAIL** |
| H1 | synthetic | capacity | 0.0259 | 0.0166 | 0.440 | 0.0094 | **PASS** |
| H2 | synthetic | matched | 0.0078 | 0.5867 | 1.948 | 0.5789 | **FAIL** |
| H2 | synthetic | capacity | 0.0078 | 0.0150 | 0.634 | 0.0072 | **PASS** |
| H1 | airpassengers | either | 0.4563 | 0.3080 | 0.388 | 0.1483 | FAIL |
| H2 | airpassengers | either | 0.2201 | 0.3112 | 0.343 | 0.0911 | FAIL |

The verdict combines a 25% relative and a 0.02 absolute tolerance. Because the absolute
one was set after seeing the numbers, it is worth stating that it is not doing the work:
the passing gaps (0.0072, 0.0094) and the failing ones (0.5346, 0.5789) are separated by a
factor of ~60, and every verdict in the table is unchanged for any absolute tolerance
between 0.01 and 0.05.

Reading:

* **Both hypotheses hold** on the synthetic base under the `capacity` rank convention —
  the only cell where independence is exact *and* MSSA has the dimensions to represent
  the signal. All four combinations land within 0.007 of each other at ε = 0, and robust
  SSA and robust MSSA within 0.009 at ε = 5%.
* **Both fail under `matched`**, and the reason is mechanical, not statistical: MSSA
  forced to r = 4 when the independent panel needs r = 8 cannot fit, so its error jumps
  to 0.56. This is decision (b) in §6.
* On the real base both fail, but the absolute gaps (0.09–0.15) sit against a
  clean-data floor of 0.22–0.31 — they mostly reflect the univariate and multivariate
  approximation floors differing at fixed r, not a robustness effect. I do not think H1/H2
  are cleanly testable on a real base at T = 144.
* Note also that even under independence, MSSA is slightly *better* than SSA
  (0.0166 vs 0.0259), not merely equal — a wider block-Hankel matrix estimates the
  subspace from more columns.

### 5.3 Forecasting

One protocol note: the truncation rank is read off the *clean* signal, which is oracle
information. It is the same oracle for every method, so the comparison is fair, and I
checked that it is not smuggling in test-period information — the rank computed from the
training window alone is identical to the rank computed from the full sample. Absolute
error levels should still be read as slightly optimistic.

**Provisional — not yet re-run under the corrected solver (§5.4).** The fit results moved
by 1.2–3.0× when that was fixed, so these numbers will move too; the direction of the
comparison is unlikely to change, but no figure here should be quoted yet.

Rolling-origin RMSE against the clean signal, multivariate cells:

| base | dependence | ε | classical | Huber | gain |
|---|---|---|---|---|---|
| synthetic | shared | 0% | 0.0231 | 0.0226 | 1.02× |
| synthetic | shared | 5% | 0.7783 | 0.1423 | **5.47×** |
| synthetic | shared | 10% | 0.9344 | 0.3058 | 3.06× |
| synthetic | independent | 5% | 1.0253 | 0.5510 | 1.86× |
| airpassengers | shared | 5% | 1.8845 | 1.1757 | 1.60× |
| airpassengers | shared | 10% | 1.9161 | 0.9988 | 1.92× |

Robustness does translate into forecast accuracy under additive contamination. At ε = 0
there is **no systematic cost, but not quite "no cost"**: across the four base ×
dependence cells the robust forecast is 2% and 10% *better* in two of them and 5% and 9%
*worse* in the other two, so the honest statement is that the tax is within noise rather
than provably zero. (On model fit, by contrast, ε = 0 parity was clean.) Two things to
flag:

* **The recurrent forecast can explode, and robust fits do it more often.** A
  near-vertical estimated subspace makes the `1/(1 − ν²)` factor in the LRR enormous; on
  the shortest training window a robust fit landed at ν² = 0.9971, dominant
  characteristic root 1.337, and a 12-step forecast ~140× the series' scale (other seeds
  reached 10⁸). Unguarded, one such origin sets the whole cell's RMSE. Origins whose
  dominant root implies >10× growth over the horizon are now rejected and counted
  instead of averaged; the rejection rate reaches 25% for Huber on AirPassengers at
  ε = 10%, against 0% for classical. Worth reporting as a property of the method.
* The classical forecast is never rejected, so on this axis robustness costs stability.

### 5.4 R cross-check — done

R is now installed and the check has run.

The first run of this check passed everywhere, and that turned out to be worthless: at the
Phase-2 contamination setting the *non-robust* SVD also sits only 0.028 from the R robust
answer, so it would have passed the same threshold. A cross-check that a wrong
implementation also passes is not evidence. The fixtures below are therefore contaminated
hard enough (10% of cells, 15× sd) that the classical control fails clearly, and the
control is reported next to every comparison.

| fixture | H shape | Huber vs `RobRSVD` | _control:_ classical vs `RobRSVD` | L1 vs `robustSvd` | _control_ |
|---|---|---|---|---|---|
| narrow | 40×42 | 0.9913 ✗ | 1.0000 | 0.9961 ✗ | 1.0000 |
| medium | 40×82 | 0.1839 ✗ | 0.9690 | 0.0696 ✓ | 0.9841 |
| wide | 40×273 | 0.0318 ✓ | 0.3003 | reference failed | — |
| **mssa-scale** | **40×805** | **0.0203 ✓** | **0.9945** | reference failed | — |

Metric is the subspace distance between leading r = 2 left subspaces; two unrelated
subspaces sit at ≈ 1.

**Validated at the scale that matters.** On the realistic block-Hankel matrix our Huber
backend lands 0.0203 from the R reference while the non-robust control sits at 0.9945 —
the test discriminates sharply and we are on the right side of it.

**But the narrow fixture is a genuine failure, and I would rather report it than bury it.**
At 40×42 both backends diverge from the reference. The cause is specific: our solver is a
joint IRLS *by imputation*, which replaces down-weighted cells with the current model's own
values — making the current model a fixed point, so an initialisation already corrupted by
outliers cannot be escaped. On that fixture the model sits at distance 1.0 from the truth
at iteration 1 and never moves, and does not converge even after 2000 sweeps. It is not a
weighting failure: the Huber weights are correct throughout (0.14 on contaminated cells vs
0.93 on clean). The R package's per-component deflation escapes the basin; ours does not.

**This is not confined to narrow matrices, and it is now fixed.** My first reading was that
the damage was bounded by matrix width (K ≥ 122 safe), which would have put all Phase-2 and
design-v2 results inside a validated region. That was wrong. Repeating the measurement at a
well-posed rank on the Phase-2 grid itself — K = 1506 — the capture is plainly present:
correcting it improves signal recovery by 1.2–3.0× at every contamination level, and 9 of
10 seeds at ε = 5% prefer the corrected start. Width changes how often it happens, not
whether it can.

The fix: neither candidate starting point is safe alone. The classical SVD is already
rotated onto the outliers when they dominate; a Winsorized start escapes that but perturbs
an exactly low-rank *clean* matrix off its exact solution and sticks there, which would
break the ε = 0 equivalence with the classical SVD that makes this comparison fair at all.
So the fit now runs from both and keeps whichever has the lower M-estimation objective,
scored at a common scale. The estimator, its weights and its fixed-point equation are
untouched — only which fixed point gets reported. On clean data ε = 0 moves by 3e-12; the
narrow fixture recovers from a median subspace distance of 0.637 to 0.076.

Two caveats on the table above, both worth knowing before it is quoted. The two seeds that
fail at 40×805 are *not* capture — their clean spectra have s₃/s₂ ≈ 0.98, so the rank-2
target subspace is degenerate and no estimator can be scored against it; the fix leaves
them bit-identical. And this whole ladder runs at r = 2 against a signal of SSA rank 6,
which is below the rank at which the robust and classical fits are guaranteed to agree even
on clean data. **The cross-check should be re-run at the corrected default, at a
well-specified rank, over several seeds** — that is the next item on my list.

Two further findings from doing the check:

1. **`RobRSVD` has no `rough` argument.** Robustness is controlled by `irobust`, which
   **defaults to FALSE**. The paper's Huber variant is
   `RobRSVD(M, irobust = TRUE, huberk = 1.345, uspar = 0, vspar = 0)`; a call without
   `irobust = TRUE` silently runs the non-robust regularized SVD. (Also: the pcaMethods
   function is `robustSvd`, not `robustSVD`, and `RobRSVD` returns the singular value as
   `s`, not `d`.) The package is archived on CRAN and had to be installed from source.
2. **`pcaMethods::robustSvd` does not survive a realistic MSSA trajectory matrix** — it
   runs to width ≈ 82 and then fails with `missing value where TRUE/FALSE needed`. Our
   Phase-2 matrices are 40×805. So the L1 reference is validated only on narrow fixtures,
   and is not a usable drop-in at MSSA scale.

---

## 6. Decisions I would like from you before the full-scale run

**(a) What does H1's "independent" mean?** Independent *generating processes* (which
phase-randomised surrogates give, at the cost of the low-rank structure), or
*empirically uncorrelated* series (which no set of real series delivers at T = 144)?
The two lead to different designs and, I suspect, different verdicts on H1.

**(b) Should MSSA be given more rank than SSA under independence?** p independent
signals jointly span several times more dimensions than any one of them alone, but
horizontal MSSA has a single L-dimensional row space for all of them. Forcing MSSA to
use the univariate r hands it a subspace too small to hold the signal it is scored on;
letting it have more rank arguably makes the comparison uneven in the other direction.
I currently report **both** conventions (`matched` and `capacity`) and they disagree
about H1. I would rather adopt whichever you consider the fair comparison.

**(c) Rank selection rule for the real-data arm.** I have used the scree elbow (r = 8)
and swept r to show the sensitivity. If the group has a standard rule, I will use it.

**(d) Is the level-shift result believable, and should we chase it?** Both methods
failing outright on level shifts (error > 1 at every setting) is the kind of clean
negative result that is either a genuine limitation of the estimators or a sign that
level shifts need a different treatment — segmentation before decomposition, or a
robust *loss on differences* rather than levels. I would like your read before I invest
in it, because if it is genuine it looks like the most promising gap in the 2020 method.

**(e) ε or event count for the structured types?** See the caveat in §5.1 — ε cannot
express a small level-shift rate.

**(f) Anything in §1 you would add, drop, or re-level** before I commit compute to the
full grid.

_(The R cross-check, which was item (d) in the draft of this note, is now done — §5.4.)_
