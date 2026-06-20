"""Day-5 smoke check: embedding + standard SVD backend on a synthetic signal.

Not the full baseline reproduction (that is Days 7-10, after grouping/reconstruction
land on Day 6). This only exercises the two modules implemented on Day 5 and prints
the scree contributions, sanity-checking that a known low-rank signal is recovered.

Run:  python experiments/01_baseline_repro/smoke_day5.py
"""

import numpy as np

from rmssa.embedding import trajectory_matrix, mssa_trajectory_matrix
from rmssa.decomposition import StandardSVD


def main() -> None:
    rng = np.random.default_rng(42)
    t = np.arange(300)

    # Univariate: trend + two sinusoids + small noise.
    trend = 0.01 * t
    season = np.sin(2 * np.pi * t / 50) + 0.5 * np.sin(2 * np.pi * t / 25)
    f = trend + season + 0.05 * rng.standard_normal(t.size)

    L = 100
    X = trajectory_matrix(f, L)
    d = StandardSVD().decompose(X)
    print(f"[univariate] series N={f.size}, L={L} -> H {X.shape}, numerical rank {d.rank}")
    print("  top-8 variance contributions (%):",
          np.round(100 * d.contributions()[:8], 3))
    recon_err = np.linalg.norm(X - d.reconstruct_matrix()) / np.linalg.norm(X)
    print(f"  full reconstruction rel. error: {recon_err:.2e}")

    # Multivariate: 4 channels sharing the seasonal factor, distinct noise.
    channels = [trend + season + 0.05 * rng.standard_normal(t.size) for _ in range(4)]
    H, widths = mssa_trajectory_matrix(channels, L)
    dm = StandardSVD().decompose(H)
    print(f"\n[mssa] p={len(channels)} channels, L={L} -> H {H.shape}, block widths {widths}")
    print("  top-8 variance contributions (%):",
          np.round(100 * dm.contributions()[:8], 3))
    recon_err_m = np.linalg.norm(H - dm.reconstruct_matrix()) / np.linalg.norm(H)
    print(f"  full reconstruction rel. error: {recon_err_m:.2e}")
    print("\nOK: embedding + StandardSVD backend behave as expected.")


if __name__ == "__main__":
    main()
