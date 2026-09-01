# Short update to Prof. Rodrigues — 2 Sep 2026

**Subject:** Robust MSSA — R validation done, and two questions

**Attach:** `report/simulation_design_v2.md` (the revised design table promised in July)

---

Dear Professor Rodrigues,

A short update, and two questions before I run the study at full scale.

**The R cross-check is done and passes.** At realistic MSSA size (40×805) my Huber
implementation sits 0.020 from the R reference, against 0.995 for a non-robust control.
Two notes for the paper: `RobRSVD`'s robustness flag is `irobust` and it defaults to
`FALSE` (there is no `rough` argument), so the obvious call silently runs the non-robust
SVD; and `pcaMethods::robustSvd` breaks above about 82 columns, so it cannot be used at
MSSA scale.

**It also exposed a fault in my own solver, now fixed.** The iteration could be captured by
the outliers at its starting point and never recover. It now runs from two starting points
and keeps the better fit. Signal recovery improves 1.2–3.0× at every contamination level,
clean-data results are unchanged, and much of the seed-to-seed scatter I mentioned in July
turns out to have been this rather than sampling noise.

**The main finding survives the fix: the robust advantage is specific to isolated additive
outliers.** It is about 1.5× against innovational outliers, 1.1× against patches, and
1.00× against level shifts — where classical and robust both fail outright. That looks like
the most interesting gap in the 2020 method.

Two questions, both of which change the design:

1. In H1, does "independent" mean independent generating processes, or empirically
   uncorrelated series? At T = 144 independent series still show mean |corr| of 0.15–0.42
   in sample.
2. Should MSSA be allowed more rank than SSA under independence? This one choice decides
   whether H1 passes, and I would rather use whichever you consider the fair comparison.

The design table is attached. I will hold the full run until I hear from you.

Thank you very much,

Ajayendra Kumar Bansod
B.Tech (Hons.), IIT Kharagpur
