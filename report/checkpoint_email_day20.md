# Draft — mid-internship checkpoint email to Prof. Rodrigues

> Draft for the intern to review/edit before sending. Plain-text below the line.

---

Subject: Robust MSSA — mid-internship checkpoint (synthetic validation complete)

Dear Professor Rodrigues,

A short update at the halfway point. Following your guidance, I implemented the two
robust SVD algorithms from Rodrigues et al. (2020, Entropy) — the Huber (RHSSA) and
L1-norm (RLSSA) variants — behind the same interchangeable decomposition interface as
the standard MSSA baseline, and completed the synthetic-validation phase against a known
low-rank signal. The full classical-vs-robust × univariate-vs-multivariate comparison is
in place.

Key findings on synthetic panels:

- No accuracy cost at zero contamination: all methods recover the clean signal to ~1%,
  so the comparison is fair. As contamination rises, classical (M)SSA collapses (recovery
  error grows past 1.0 — i.e. worse than a zero prediction — around 10–20% contamination)
  while Robust MSSA degrades gracefully — e.g. at 2% contamination Robust MSSA is ~27×
  more accurate than classical MSSA.
- Multivariate consistently beats univariate under contamination.
- Robust MSSA's advantage is largest with longer windows and longer series, and smallest
  with short windows and higher-rank signals; it wins in every regime I tested.
- The Huber and L1 variants are almost equivalent in accuracy (their factor subspaces
  differ negligibly even at 20% contamination), but Huber converges faster — so I am using
  Huber as the primary and keeping L1 as the second reported algorithm.

Two small points I'd value your view on:

1. I reproduced the estimators faithfully (including the Huber d = 1.345), but solve the
   robust low-rank step with a joint iteratively-reweighted scheme rather than the exact R
   packages (pcaMethods::robustSVD, RobRSVD). I've scripted a numerical cross-check against
   those packages but don't have R set up locally — would you like me to run it against the
   originals, or is the substitution acceptable for our purposes?
2. Please confirm you're happy carrying both algorithms (Huber primary, L1 secondary) into
   the empirical phase.

The code and a short synthetic-validation report are on GitHub
(https://github.com/Ajayendra2705/robust-mssa, tag v0.2-robust-synthetic); 99 tests pass
and every figure reproduces from a script + seed. Next I'll move to the empirical study on
the equity and macro panels.

Thank you very much.

Best regards,
Ajayendra Kumar Bansod
B.Tech (Hons.), IIT Kharagpur
