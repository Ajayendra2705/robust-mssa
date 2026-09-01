# Short update to Prof. Rodrigues — 2 Sep 2026

**Subject:** Robust MSSA — R validation done, a solver fault it uncovered, and two questions

**Attach:** `report/simulation_design_v2.md` (the revised design table promised in July)

---

Dear Professor Rodrigues,

A short update since my July reply, and two questions where I would value your steer
before running the study at full scale.

**The R cross-check is done.** I installed R and ran both backends against `RobRSVD` and
`pcaMethods::robustSvd` on the same matrices. At realistic MSSA size (40×805) my Huber
implementation lands 0.020 from the R reference while a non-robust control sits at 0.995 —
so the test discriminates, and we pass it. Two things worth recording for the paper:
`RobRSVD` has no `rough` argument (robustness is `irobust`, which defaults to `FALSE`, so a
call without it silently runs the *non-robust* regularized SVD), and
`pcaMethods::robustSvd` fails above about 82 columns, so it cannot serve as a drop-in
reference at MSSA scale.

**It also uncovered a fault in my own solver, which is now fixed.** The iteration is a
fixed-point scheme rather than a descent method, so it can be captured by the outliers at
the starting point and never leave. Repeating the check over 10 seeds instead of one shows
this happening at every matrix size — including 2 of 10 seeds at the size that passed
above. The fix is to run the iteration from both the classical start and a Winsorized one
and keep whichever has the lower objective, scored at a common scale; the estimator itself
is untouched. Model-fit results improve by 1.2–3.0× at every contamination level, ε = 0 is
unchanged to twelve digits (so the comparison stays fair on clean data), and the wide
seed-to-seed scatter I flagged in July turns out to have been largely this rather than
sampling noise — the spread at 5% contamination falls from 8.1× to 2.2×.

Because of that, the contamination-type and forecasting tables need re-running before I
quote numbers. The qualitative result I most want to show you should survive it: **the
robust advantage is specific to isolated additive outliers**. Against patches of
consecutive outliers it falls to about 1.1×, and against level shifts to exactly 1.0× —
with both classical and robust failing outright. If that holds, it looks like the most
interesting gap in the 2020 method.

Two questions, both of which change the design:

1. In H1, does "independent" mean independent generating processes, or empirically
   uncorrelated series? They differ at these sample sizes — independent but strongly
   autocorrelated series still show mean |corr| ≈ 0.35 at T = 144.
2. Should MSSA be allowed more rank than SSA under independence? This single choice decides
   whether H1 passes or fails, and I would rather adopt whichever you consider the fair
   comparison than pick one myself.

The revised simulation design table is attached. I will hold the full-scale run until I
hear from you.

Thank you very much,

Ajayendra Kumar Bansod
B.Tech (Hons.), IIT Kharagpur
