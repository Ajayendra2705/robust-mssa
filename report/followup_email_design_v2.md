# Follow-up to Prof. Rodrigues — revised simulation design + first results

**Subject:** Robust MSSA — revised simulation design, and one result that changes the story

**Attach:** `report/simulation_design_v2.md` (the design table and full detail)

> ### ⚠️ NEVER SENT — superseded, and the wrong register anyway
>
> Drafted 10 Aug 2026, superseded by the shorter `update_email_sep.md` before sending.
> That shorter version was itself challenged by the supervisor on 3 Sep as AI-generated;
> this one is longer and would have failed harder. Kept only as a record.
>
> Its numbers also predate the 2 Sep initialisation fix and several are now wrong —
> `results_robust_init.md` supersedes it. Nothing here should be quoted.

---

Dear Professor Rodrigues,

Following up on my reply of 25 July. I have built the revised simulation design we
discussed and run it at reduced replication, so this note carries the design table I
promised plus the first results — including one that I think changes what we should be
claiming.

**The R cross-check is done, and it found something.** I installed R locally and ran both
backends against the original functions.

My first version of this check passed everywhere — and was worthless. At our Phase-2
contamination setting the *non-robust* SVD also sits only 0.028 from the R robust answer,
so a completely wrong implementation would have passed the same threshold. I redid it with
contamination heavy enough that the non-robust control fails clearly, and report the
control beside every comparison.

On that harder test: at the realistic MSSA scale (40×805) our Huber backend lands **0.0203
from the R reference while the non-robust control sits at 0.9945** — so the check now
discriminates, and we pass it. It also passes at 40×273.

It does **not** pass on a narrow 40×42 matrix, and I think that is worth reporting rather
than burying. Our solver is a joint IRLS *by imputation*: down-weighted cells are replaced
with the current model's own values, which makes the current model a fixed point, so an
initialisation already corrupted by outliers cannot be escaped. On that fixture it sits at
distance 1.0 from the truth at iteration 1 and never moves, and fails to converge after
2000 sweeps — while the Huber weights themselves are perfectly sensible throughout (0.14 on
contaminated cells, 0.93 on clean ones). `RobRSVD`'s per-component deflation escapes the
basin; ours does not. I have mapped the boundary: below about 120 trajectory columns the
solver degrades, above it recovers the true subspace to 0.05–0.11 and beats the classical
SVD everywhere I tested. Every matrix in the Phase-2 and current experiments has ≥ 400
columns, so those results are inside the validated region — but a robust initialisation is
an obvious improvement and I would like to add it.

Two further things surfaced that I think you will want to know:

1. **`RobRSVD` has no `rough` argument.** Robustness is controlled by `irobust`, which
   defaults to `FALSE`. The paper's Huber variant is
   `RobRSVD(M, irobust = TRUE, huberk = 1.345, uspar = 0, vspar = 0)` — a call without
   `irobust = TRUE` silently runs the *non-robust* regularized SVD. The package is also
   now archived on CRAN and has to be installed from source.
2. **`pcaMethods::robustSvd` cannot handle a realistic MSSA trajectory matrix.** It runs
   up to about 82 columns and then fails outright. Our block-Hankel matrices are 40×805.
   So the L1 reference is validated only on narrow fixtures, and is not usable as a
   drop-in at MSSA scale.

**The result that changes the story.** Extending the contamination model from isolated
additive outliers to the four standard types shows that our Phase-2 headline was
specific to the one type we happened to test. Robust MSSA (Huber) vs classical MSSA,
AirPassengers-based panel, as a ratio of signal-recovery errors:

Ratios are net of the approximation floor — with a real base series at r = 8 nothing can
beat 0.144 on clean data, and a raw ratio understates a robust fit already sitting on that
floor. (Raw ratios in the attached note; the ordering is the same.) These are not directly
comparable with the Phase-2 27×, which was a raw ratio on a synthetic panel against a floor
of 0.01 — what is controlled here is the comparison *between types*, which shares a panel,
a rank and a floor.

| outlier type, 8× sd | ε = 1% | ε = 5% | ε = 10% | ε = 20% |
|---|---|---|---|---|
| additive | ~9× | ~7× | ~2.5× | ~1.9× |
| innovational | 1.9× | 1.5× | 1.4× | 1.4× |
| patches | 1.1× | 1.1× | 1.0× | 1.0× |
| level shifts | 1.0× | 1.0× | 1.0× | 1.0× |

I have deliberately rounded the additive row, because with only 3 seeds per cell it is not
pinned down — the per-seed gains at ε = 5% are 25.6, 16.4 and 4.0. The structured rows, by
contrast, are almost noiseless (patch 1.06/1.09/1.05; level shift 1.00/1.01/1.00), and the
*ordering* is cleanly separated: at every rate the worst additive seed still beats the best
patch and level-shift seed. So I would put weight on the qualitative result and none yet on
the additive multiplier.

I also repeated the whole comparison on four different clean signals, since seeds in this
design are paired on the signal and cannot test that:

| clean signal | clean floor | additive | innovational | patch | level shift |
|---|---|---|---|---|---|
| AirPassengers + co2 | 0.1437 | 7.01× | 1.49× | 1.07× | 1.01× |
| co2 + sunspots | 0.3176 | 5.95× | 1.87× | 1.29× | 1.00× |
| sunspots + co2 | 0.2924 | 7.54× | 1.81× | 1.22× | 1.00× |
| synthetic | 0.0103 | 8.10× | 1.50× | 1.19× | 1.00× |

The ordering is identical in all four, over floors spanning a factor of 30. The last row
is the one that settles it: on the synthetic signal the approximation floor is 0.0103 —
essentially the Phase-2 regime, where nothing is masked — and additive contamination still
gives 8.1× while level shifts give exactly 1.00×. So this is not a floor artefact, not a
peculiarity of AirPassengers, and not a seed accident.

Against patches the robust gain collapses to nothing, and against level shifts it is
exactly nothing — with *both* methods failing outright (recovery error 1.1–4.0, worse
than predicting zero).

For level shifts I can show the mechanism rather than just assert it. At ε = 5% the
planted outlier energy is ‖O‖_F/‖S‖_F = 3.027 and the observed recovery error is 3.006 —
the reconstruction retains essentially the *whole* shift and rejects none of it. Under
additive contamination at ε = 20% the outlier energy is 3.575 against a robust error of
0.84, so there the energy plainly is being rejected. A permanent step is simply not an
outlier to a per-cell M-estimator: within its own column it is 50% contamination, well
past breakdown, and it ends up absorbed into the trend. A patch is the same story in
miniature — locally it looks like signal.

One thing I should flag because my first reading of it was wrong: the outlier *magnitude*
turns out to matter much less than it appears. On raw ratios the additive gain looks like
it halves going from 8× to 3× sd, but almost all of that is the floor compressing the
ratio; net of the floor it is 7.0× vs 6.3×, and at higher contamination the smaller
outliers actually score slightly better. What does drive the gain is the contamination
rate, which is ordinary breakdown behaviour.

If this holds at full scale, I think it is a better thing to report than the 27× figure,
and it points fairly directly at where a methodological contribution might sit.

**Your two hypotheses.** Both now testable, and both hold — but only in the regime where
the question is well posed. On the synthetic base, where independence is exact and the
signal rank is exact, all four combinations land within 0.007 of each other at ε = 0
(H2), and robust SSA and robust MSSA within 0.009 at ε = 5% (H1). They fail when
multivariate MSSA is forced to use the univariate rank, which is a mechanical artefact:
`p` independent signals need several times more dimensions jointly than any one needs
alone, but horizontal MSSA has one `L`-dimensional row space for all of them. Worth
noting that even under independence MSSA is slightly *better* than SSA, not merely
equal, since a wider block-Hankel matrix estimates the subspace from more columns.

**Forecasting is now in the design**, scored on the same 2×2 grid: rolling origin,
h = 1…12, errors against the clean signal. Robustness does carry over — 5.5× better RMSE
at ε = 5% on the synthetic panel and ~1.6–1.9× on AirPassengers. At ε = 0 there is no
*systematic* cost, though unlike the model-fit case I would not claim it is exactly zero:
across the four base × dependence cells the robust forecast is 2% and 10% better in two
and 5% and 9% worse in the other two.

One caveat worth recording: the recurrent forecast can explode when the estimated subspace
is near-vertical, and robust fits do this materially more often than classical ones — 25%
of origins on AirPassengers at ε = 10% (and 24% pooled for the univariate robust cells),
against **0 of 72** for classical anywhere. I now detect and reject those origins rather
than averaging a 10⁸ forecast into the table, and report the rejection rate as an outcome
in its own right.

**Where I would like your steer** (fuller versions in §6 of the attached note):

1. **What does "independent" mean in H1** — independent generating processes, or
   empirically uncorrelated series? They are not the same thing at T = 144: independent
   but strongly autocorrelated series still show mean |corr| ≈ 0.35 in sample. Relatedly,
   I could not find any construction giving real series that are simultaneously
   low-SSA-rank, mutually independent, and available in arbitrary number — phase-
   randomised surrogates buy independence but destroy the low rank (AirPassengers needs
   r = 10 for 99% of trajectory variance; its surrogate needs r = 28).
2. **Should MSSA be allowed more rank than SSA under independence?** This single choice
   decides whether H1 passes or fails. I report both conventions and would rather adopt
   whichever you consider the fair comparison.
3. **Is the level-shift failure worth chasing?** It is a clean negative result and looks
   like the most promising gap in the 2020 method, but I would rather have your read
   before investing in it.

Everything is implemented and tested (187 passing tests) and the code is on the repo. I
have held back the full-scale run until I hear from you, since items 1 and 2 change what
the design should be.

Thank you very much,

Ajayendra Kumar Bansod
B.Tech (Hons.), IIT Kharagpur
