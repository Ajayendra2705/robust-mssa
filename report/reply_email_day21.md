# Reply to Prof. Rodrigues — simulation-study design questions (Day 21)

**Subject:** Re: Robust MSSA — mid-internship checkpoint (simulation design details)

---

Dear Professor Rodrigues,

Thank you for the feedback and for the questions — they point at exactly the places where
the current design is thin, so let me answer them precisely and then say how I propose to
extend it.

**What the simulation study currently is**

1. **Series.** So far, entirely artificial panels, not real series. Each panel is
   `X = S + N + O`: the clean signal `S` is `p` series formed as random loading
   combinations of `k` smooth latent factors (factor 1 = sine of period 50 plus a linear
   trend, further factors = sines of distinct periods drawn from {50, 30, 80, 20, 65, 40}
   so they never alias), normalised to unit standard deviation; `N` is i.i.d. Gaussian
   noise with sd 0.03, i.e. about 3% of the signal scale. The reason for artificial data
   was that it gives an exactly known clean signal and an exactly known factor subspace,
   which is what let me score recovery in absolute terms. The real data (a 2005–2024
   equity panel and a FRED macro panel) is downloaded but has been reserved for the
   empirical phase and has not been contaminated.

2. **Contamination.** One type only: isolated additive outliers, placed uniformly at
   random over the `T × p` cells, random sign, magnitude 8× the signal standard
   deviation. Percentages ε ∈ {0, 1, 2, 5, 10, 20}% of cells, each averaged over 5 seeds.

3. **Number of variables.** `p = 6` in the main grid, swept over p ∈ {3, 6, 10, 15} in the
   dimension study. Alongside it: T ∈ {150, 300, 600} (baseline 300), window
   L ∈ {20, 40, 60, 80} (baseline 50), k ∈ {1, 2, 3, 4} factors, with the retained rank
   set to r = 2k + 2 to cover the signal's SSA-rank.

4. **Relationship between the variables.** Yes, and this is the main limitation: the
   series are *strongly* related by construction — all `p` of them load on the same `k`
   common factors, so the clean panel is exactly rank-`k`. I have not yet run the
   independent case at all, which means your expectation (robust SSA ≈ robust MSSA for
   independent series) is not testable on the current design.

5. **Fit or forecasting.** Model fit only. Two ground-truth metrics per configuration:
   signal-recovery error ‖Ŝ − S‖_F / ‖S‖_F, and subspace-recovery error measured as the
   largest principal angle between the estimated and the true factor subspace. No
   forecasting comparison yet — the robust forecasting algorithm from the 2020 paper was
   scheduled for the following phase.

**How I propose to extend it, following your suggestions**

- **Base series.** Add real benchmark series as the signal, as you suggest: AirPassengers
  as the primary one, plus a small bank of classics with different dynamics (e.g. Nile,
  sunspots, a CO₂/gas series). The uncontaminated original series is treated as the known
  clean signal, so ground-truth scoring is preserved. The artificial generator stays as a
  controlled complement where rank and factor structure need to be set exactly.
- **Dependence as an explicit design factor**, with three levels: (i) *independent* — each
  series driven by its own unrelated base series/factor; (ii) *partially shared* — common
  factors plus idiosyncratic ones, with a tunable shared proportion; (iii) *fully shared*
  — the current setting. This turns your two expectations into hypotheses the study
  actually tests: **H1** — robust SSA ≈ robust MSSA when the series are independent;
  **H2** — all four combinations coincide when there is no contamination *and* the
  variables are independent. Both are also useful sanity checks on the implementation.
- **Contamination types**, beyond the isolated additive case: patches of consecutive
  outliers, level shifts, and innovational outliers, each at the same ε grid; and outlier
  magnitude varied over {3, 5, 8}× sd, since 8× is generous to the robust estimators and
  the interesting regime is likely the smaller ones.
- **Forecasting arm.** Score every cell of the classical-vs-robust × univariate-vs-
  multivariate grid on out-of-sample forecasts as well as on fit — rolling origin, horizons
  h = 1…12, RMSE and MAE against the clean signal — using the robust forecasting algorithm
  from the 2020 paper. This makes fit and forecast directly comparable within one design.

**On the validation against the original functions:** understood, I will do that rather
than rely on the substitution. I will install R locally and check my Huber and L1
implementations against `RobRSVD` and `pcaMethods::robustSVD` on the same inputs, and
report the numerical deviation on a few of the results already produced.

And noted on the L1 results going to the appendix — I will keep reporting both, with Huber
as the primary.

I will send the revised simulation design as a short table before running it at full scale,
in case you want anything changed first.

Thank you very much for the guidance.

Best regards,
Ajayendra Kumar Bansod
B.Tech (Hons.), IIT Kharagpur
