# Cross-check against the original R implementations

Our two robust SVD backends run against the reference implementations on the same fixture matrices. The metric is the subspace distance (sine of the largest principal angle) between the leading r=2 left singular subspaces — the rotation-, sign- and ordering-invariant way to ask whether two implementations found the same factor subspace.

| fixture | H shape | RHSSA (Huber) vs `RobRSVD` | _control:_ classical vs `RobRSVD` | RLSSA (L1) vs `robustSvd` | _control:_ classical vs `robustSvd` |
|---------|---------|----------------------------|-----------------------------------|---------------------------|-------------------------------------|
| narrow | 40x42 | 0.9913 ✗ | 1.0000 ✗ | 0.9961 ✗ | 1.0000 ✗ |
| medium | 40x82 | 0.1839 ✗ | 0.9690 ✗ | 0.0696 ✓ | 0.9841 ✗ |
| wide | 40x273 | 0.0318 ✓ | 0.3003 ✗ | reference failed | — |
| mssa-scale | 40x805 | 0.0203 ✓ | 0.9945 ✗ | reference failed | — |

Agreement threshold: subspace distance < 0.1. For scale: two *unrelated* rank-2 subspaces in R^40 sit at distance ≈ 1.

## Why the control columns are there

Fixtures are contaminated at 10% of cells with 15× outliers **specifically so that this test can fail**. At the gentler Phase-2 setting (5% at 8× sd) the plain non-robust SVD sits only 0.028 from the R robust answer — it would pass a 0.10 threshold as well, and agreement would be evidence of nothing. At the setting used here the classical subspace is destroyed (distance ≈ 1 from the truth, and ≈ 1 from the R reference) while a correct robust fit stays near 0.02. The control column is the reading that makes the main column mean something: our backends land next to the R references, and a non-robust implementation lands nowhere near them.

## Verdict: validated at MSSA scale, with a real limit on narrow matrices

**At the size that matters we pass a test that can fail.** On the realistic block-Hankel fixture (40×805) our Huber backend lands 0.0203 from the R reference while the non-robust control sits at 0.9945 — the test discriminates sharply and we are on the right side of it. The 40×273 fixture agrees too (0.0318, control 0.3003), and L1 passes at 40×82 (0.0696, control 0.9841).

**The narrow fixture (40×42) is a genuine failure and should be stated plainly.** Both backends diverge from the reference there (≈0.99). Diagnosis: our solver is a joint IRLS-*by-imputation*, which replaces down-weighted cells with the *current* model's own values. That makes the current model a fixed point, so an initialisation already corrupted by outliers cannot be escaped — on this fixture the model sits at distance 1.0 from the truth at iteration 1 and never moves, and the iteration fails to converge even after 2000 sweeps. Notably this is *not* a weighting failure: the Huber weights are correct throughout (mean 0.14 on contaminated cells against 0.93 on clean ones). The R package's per-component deflation escapes the basin; ours does not.

**Measured validity domain** (r=2, distance to the true subspace, 3 seeds): the failure is confined to the narrowest matrices. At K=42 with 10% contamination the solver degrades (0.40 at 8× outliers, 0.69 at 15×); at **K ≥ 122 it recovers the true subspace to 0.048–0.107 and beats the classical SVD in every cell tested**. Every trajectory matrix used in the Phase-2 and design-v2 experiments has K ≥ 400, so those results sit well inside the validated region. A robust initialisation (median-based or subsampled) is the obvious fix and should be done before any short-window or narrow-panel work.

> ### ⚠️ Superseded — see `results_robust_init.md`
>
> The diagnosis above (imputation makes the current model a fixed point) is right; the
> **validity domain is not**. Width is not the controlling variable. The Phase-2 grid
> itself — K = 1506, well inside the supposedly safe region — is affected: fixing the
> initialisation improves signal recovery there by 1.2–3.0× at every contamination level,
> with 9 of 10 seeds at ε = 5% preferring the new start. So "K ≥ 122 is safe" was too
> generous, and the Phase-2 results were *not* fully inside a validated region.
>
> (The 2/10 seeds that fail at K = 805 on the fixture ladder above are a *different*
> problem and are not evidence of capture: their clean spectra have s₃/s₂ ≈ 0.98, so the
> rank-2 target is degenerate and the fix leaves them bit-identical. This whole ladder
> also runs at r = 2 against a rank-6 signal — see §4 of `results_robust_init.md`.)
>
> The fix is in: the backends now run the iteration from both the classical and a
> Winsorized start and keep whichever has the lower M-estimation objective at a common
> scale (`init="auto"`, the default; `init="classical"` reproduces everything below).
> A Winsorized start *alone* is not a fix — it breaks the clean-data equivalence with the
> classical SVD, which is what makes the ε = 0 comparison fair. With the two-start default
> the narrow fixture recovers (0.637 → 0.076 median distance, 6/10 → 1/10 failures) and
> the Phase-2 numbers improve 1.2–3.0× at every contamination level while ε = 0 stays
> identical to twelve digits.
>
> **This cross-check should be re-run against R at the corrected default**, and at more
> than one seed per fixture. The pass/fail verdicts below are for the old
> `init="classical"` behaviour.

## Two things worth recording

**1. The reference call had to be corrected.** `RobRSVD` has no `rough` argument; robustness is controlled by `irobust`, which **defaults to FALSE**. The paper's Huber variant is `RobRSVD(M, irobust = TRUE, huberk = 1.345, uspar = 0, vspar = 0)`. A call without `irobust = TRUE` silently runs the *non-robust* regularized SVD — so this correction is the difference between validating the right algorithm and validating the wrong one. (The pcaMethods function is also `robustSvd`, not `robustSVD`; and RobRSVD returns the singular value as `s`, not `d`.)

**2. `pcaMethods::robustSvd` does not survive a realistic MSSA trajectory matrix.** It runs at width K ≤ ~82 and fails above that with `missing value where TRUE/FALSE needed`. The Phase-2 block-Hankel matrices are 40×805. So the L1 reference is not a usable drop-in for MSSA at scale — which is itself part of the answer to why this project carries its own solver rather than a thin wrapper around the R packages.

_Environment: R 4.6.1; `RobRSVD` 1.0 installed from the CRAN archive (it has been archived and is not available for R ≥ 4.x through the normal channel); `pcaMethods` via Bioconductor; `matrixStats` required by `robustSvd`._
