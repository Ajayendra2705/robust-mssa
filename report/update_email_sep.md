# Short update to Prof. Rodrigues — 2 Sep 2026

**Subject:** Robust MSSA — R validation done, a solver fault it uncovered, and two questions

**Attach:** `report/simulation_design_v2.md` (the revised design table promised in July)

---

Dear Professor Rodrigues,

A short update since my July reply, and two questions where I would value your steer
before running the study at full scale.

**The R cross-check is done.** I installed R and ran both backends against `RobRSVD` and
`pcaMethods::robustSvd` on the same matrices. At realistic MSSA size (40×805) my Huber
implementation lands 0.020 from the R reference while a non-robust control sits at 0.995,
so the test discriminates and we pass it; the L1 backend agrees to 0.021. Two things worth
recording for the paper: `RobRSVD` has no `rough` argument — robustness is `irobust`, which
defaults to `FALSE`, so the obvious call silently runs the *non-robust* regularized SVD —
and `pcaMethods::robustSvd` fails above about 82 columns (it runs at 40×82 and breaks at
40×90), so it cannot serve as a drop-in reference at MSSA scale.

**It also uncovered a fault in my own solver, which is now fixed.** The iteration is a
fixed-point scheme rather than a descent method, so it can be captured by the outliers at
the starting point and never leave. I had assumed this was confined to narrow matrices;
it is not. The Phase-2 grid itself is affected, and fixing it improves signal recovery
there by 1.2–3.0× at every contamination level, with ε = 0 unchanged to within 3e-12, so
the comparison stays fair on clean data. The fix is to run the iteration from both the
classical start and a Winsorized one and keep whichever has the lower objective, scored at
a common scale; the estimator itself is untouched, and the previous behaviour is still
reproducible for comparison. The wide seed-to-seed scatter I flagged in July was largely
this rather than sampling noise — the spread at 5% contamination falls from 8.1× to 2.2×.

I have re-run the contamination-type comparison under the fix, and the result I most want
to show you survives it: **the robust advantage is specific to isolated additive
outliers.** Against innovational outliers it is about 1.5×, against patches of consecutive
outliers about 1.1×, and against level shifts 1.00× — with both the classical and the
robust fit failing outright there, recovery error around 3.0 against a clean-data floor of
0.14, i.e. both worse than predicting zero. If this holds at full replication, it looks
like the most interesting gap in the 2020 method. The forecasting comparison still needs
re-running under the corrected solver before I quote anything from it.

Two questions, both of which change the design:

1. In H1, does "independent" mean independent generating processes, or empirically
   uncorrelated series? They are not the same at these sample sizes — independent but
   strongly autocorrelated series still show mean |corr| between 0.15 and 0.42 in sample
   at T = 144, depending on how the independence is constructed.
2. Should MSSA be allowed more rank than SSA under independence? This single choice decides
   whether H1 passes or fails, and I would rather adopt whichever you consider the fair
   comparison than pick one myself.

The revised simulation design table is attached. I will hold the full-scale run until I
hear from you.

Thank you very much,

Ajayendra Kumar Bansod
B.Tech (Hons.), IIT Kharagpur
