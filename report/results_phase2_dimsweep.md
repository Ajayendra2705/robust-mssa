# Phase 2 — Day 18: sweep over factor structure & dimensions

**Experiment:** `experiments/02_synthetic_validation/run_dimsweep.py`
**Config:** `experiments/configs/dimsweep_synthetic.yaml`
**Design:** one-factor-at-a-time (OFAT) around a baseline (T=300, p=6, k=2, L=50,
noise_sd=0.03), at fixed contamination **ε = 10%**, 3 seeds. Compares **classical MSSA**
vs **Robust MSSA (Huber)**, multivariate. Rank tracks the signal SSA-rank **r = 2k + 2**;
distinct factor periods so factors never alias.
**Headline metric:** robustness gain = classical recovery error / robust recovery error
(higher = Robust MSSA worth more).

![dimension sweep](../experiments/02_synthetic_validation/outputs_dimsweep/dimsweep_recovery.png)

## Gain by dimension

**Vary k (number of shared factors), r = 2k+2**

| k | r | classical | robust | gain |
|---|---|---|---|---|
| 1 | 4 | 0.636 | 0.012 | **53.5×** |
| 2 | 6 | 0.829 | 0.064 | 13.0× |
| 3 | 8 | 0.977 | 0.095 | 10.2× |
| 4 | 10 | 1.063 | 0.186 | 5.7× |

**Vary p (panel width)**

| p | classical | robust | gain |
|---|---|---|---|
| 3 | 0.837 | 0.026 | 32.8× |
| 6 | 0.829 | 0.064 | 13.0× |
| 10 | 0.785 | 0.054 | 14.4× |
| 15 | 0.784 | 0.067 | 11.8× |

**Vary T (series length)**

| T | classical | robust | gain |
|---|---|---|---|
| 150 | 0.900 | 0.062 | 14.5× |
| 300 | 0.829 | 0.064 | 13.0× |
| 600 | 0.806 | 0.024 | 33.9× |

**Vary L (window length)**

| L | classical | robust | gain |
|---|---|---|---|
| 20 | 1.270 | 0.444 | 2.9× |
| 40 | 0.913 | 0.087 | 10.5× |
| 60 | 0.752 | 0.017 | 45.6× |
| 80 | 0.638 | 0.009 | **68.4×** |

## Where Robust MSSA helps most / least

1. **Window length L is the strongest lever.** Gain rises from 2.9× (L=20) to 68× (L=80).
   With a short window the trajectory columns are too short to resolve the signal's
   periodicities (periods up to 50) and its trend, so *both* methods struggle (classical
   1.27, robust 0.44) — but robust still wins. Large L gives robust a clean subspace to
   defend while classical stays corrupted. **L should be comfortably larger than the
   longest signal period** (an L-selection input for Day 21).
2. **Longer series (T) widen the edge** (14× → 34× from T=150 to 600): more observations
   let the robust fit pin the factor subspace down while a fixed outlier fraction is
   diluted; classical barely improves because outliers keep rotating its L2 subspace.
3. **More factors (k) shrink the edge** (53× → 5.7× from k=1 to 4): a higher-rank signal
   is harder to recover for everyone, and the robust advantage compresses — though it is
   still a 5.7× error reduction at k=4.
4. **Panel width (p) matters least** here: gain stays ~12–33× across p ∈ {3..15}. Wider
   panels neither clearly help nor hurt the *relative* advantage at this contamination.

**Takeaway:** Robust MSSA dominates classical MSSA (gain ≥ 2.9×) in *every* regime tested;
its advantage is largest with long windows, long series, few factors, and narrow panels,
and smallest with short windows and high-rank signals.

## Notes

- **Bug caught & fixed:** the L-sweep initially reported identical numbers for all L — a
  key mismatch (base config stores the window under `window`, the sweep dimension is `L`)
  silently held L=50. Fixed with an explicit dim→param mapping; results above are post-fix.
- **Subspace metric** is recorded in the CSV but is noisy at 3 seeds (largest-principal-
  angle saturates); the recovery error drives the regime conclusions here.
- Reproduce: `python experiments/02_synthetic_validation/run_dimsweep.py
  --config experiments/configs/dimsweep_synthetic.yaml`.

**Next (Day 19):** cross-check the two robust algorithms (Huber vs L1) — agreement at ε=0,
divergence pattern as ε and regime vary, per-algorithm cost/convergence.
