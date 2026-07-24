"""Day-19 comparison of the two robust algorithms: RHSSA (Huber) vs RLSSA (L1).

For each contamination level and seed, fits both robust backends (multivariate) and
records, per algorithm:
  * recovery_error (vs clean S),
  * n_iter and whether it converged,
  * wall-clock fit time;
and, per (eps, seed), the **inter-algorithm divergence** = sin of the largest principal
angle between the two algorithms' leading factor subspaces (0 = identical).

Confirms: they agree at eps=0 (both reduce to the SVD), diverge as eps grows, and have
comparable-but-distinct cost/convergence profiles.

    python experiments/02_synthetic_validation/run_algo_compare.py \
        --config experiments/configs/algo_compare.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rmssa.mssa import MSSA  # noqa: E402
from rmssa.datasets import make_synthetic_panel  # noqa: E402
from rmssa.metrics import signal_recovery_error, subspace_distance  # noqa: E402
from _grid_common import make_backends  # noqa: E402

ALGOS = ["RHSSA_huber", "RLSSA_l1"]


def fit_multivariate(X, clean_signal, backend, L, r):
    """Fit MSSA with a robust backend; return (recovery_error, U, n_iter, converged, secs)."""
    channels = [X[:, j] for j in range(X.shape[1])]
    t0 = time.perf_counter()
    model = MSSA(window=L, backend=backend).fit(channels)
    secs = time.perf_counter() - t0
    rec = np.asarray(model.reconstruct_full()).T
    err = signal_recovery_error(rec, clean_signal)
    return err, model.decomposition.U[:, :r], backend.n_iter_, backend.converged_, secs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/algo_compare.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    pan, ms = cfg["panel"], cfg["mssa"]
    L, r = ms["window"], ms["rank"]
    seeds, eps_list = cfg["seeds"], cfg["contamination"]
    sv = cfg["solver"]

    per_algo = []   # rows: eps, algo, seed, recovery_error, n_iter, converged, secs
    divergence = []  # rows: eps, seed, subspace_divergence
    for eps in eps_list:
        for seed in seeds:
            sp = make_synthetic_panel(
                T=pan["T"], p=pan["p"], k=pan["k"], noise_sd=pan["noise_sd"],
                contamination=eps, outlier_scale=pan["outlier_scale"], seed=seed,
            )
            Us = {}
            for algo in ALGOS:
                backend = make_backends(sv["max_iter"], sv["tol"])[algo](r)
                err, U, n_iter, conv, secs = fit_multivariate(sp.X, sp.signal, backend, L, r)
                Us[algo] = U
                per_algo.append(dict(eps=eps, algo=algo, seed=seed, recovery_error=err,
                                     n_iter=n_iter, converged=conv, secs=secs))
            divergence.append(dict(eps=eps, seed=seed,
                                   subspace_divergence=subspace_distance(Us["RHSSA_huber"], Us["RLSSA_l1"])))
        print(f"  eps={eps} done")

    # ---- CSVs
    (out / "algo_compare_per_algo.csv").write_text(
        "eps,algo,seed,recovery_error,n_iter,converged,secs\n"
        + "\n".join(f"{x['eps']},{x['algo']},{x['seed']},{x['recovery_error']:.6f},"
                    f"{x['n_iter']},{int(x['converged'])},{x['secs']:.4f}" for x in per_algo)
        + "\n"
    )
    (out / "algo_compare_divergence.csv").write_text(
        "eps,seed,subspace_divergence\n"
        + "\n".join(f"{x['eps']},{x['seed']},{x['subspace_divergence']:.6f}" for x in divergence)
        + "\n"
    )

    def amean(field, algo, eps):
        vals = [x[field] for x in per_algo if x["algo"] == algo and x["eps"] == eps]
        return float(np.mean(vals))

    def dmean(eps):
        return float(np.mean([x["subspace_divergence"] for x in divergence if x["eps"] == eps]))

    # ---- 3-panel plot
    colors = {"RHSSA_huber": "#1f77b4", "RLSSA_l1": "#d62728"}
    xs = [100 * e for e in eps_list]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for algo in ALGOS:
        ax[0].plot(xs, [amean("recovery_error", algo, e) for e in eps_list], "o-",
                   color=colors[algo], label=algo)
        ax[2].plot(xs, [amean("n_iter", algo, e) for e in eps_list], "o-",
                   color=colors[algo], label=algo)
    ax[0].set_title("recovery error")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=9)
    ax[1].plot(xs, [dmean(e) for e in eps_list], "o-", color="#2ca02c")
    ax[1].set_title("inter-algorithm subspace divergence")
    ax[2].set_title("iterations to converge")
    ax[2].legend(fontsize=9)
    for a in ax:
        a.set_xlabel("contamination epsilon (%)")
        a.grid(True, alpha=0.3, which="both")
    fig.suptitle(f"Day-19 Huber vs L1 (L={L}, r={r}, {len(seeds)} seeds)")
    fig.tight_layout()
    fig.savefig(out / "algo_compare.png", dpi=130)

    # ---- checks
    eps0, eps_hi = min(eps_list), max(eps_list)
    checks = {
        "agree_at_clean": bool(dmean(eps0) < 0.02),
        "both_recover_clean": bool(amean("recovery_error", "RHSSA_huber", eps0) < 0.05
                                   and amean("recovery_error", "RLSSA_l1", eps0) < 0.05),
        "diverge_as_eps_grows": bool(dmean(eps_hi) > dmean(eps0)),
    }

    summary = {
        "config": cfg,
        "per_eps": {
            str(e): {
                "divergence": round(dmean(e), 5),
                **{algo: {
                    "recovery_error": round(amean("recovery_error", algo, e), 5),
                    "n_iter": round(amean("n_iter", algo, e), 1),
                    "secs": round(amean("secs", algo, e), 3),
                } for algo in ALGOS},
            } for e in eps_list
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    (out / "algo_compare_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- console report
    print(f"\n[algo-compare] L={L} r={r} seeds={seeds}")
    print(f"{'eps':>6} {'diverge':>8} | {'hub_err':>8} {'l1_err':>8} | "
          f"{'hub_it':>6} {'l1_it':>6} | {'hub_s':>6} {'l1_s':>6}")
    for e in eps_list:
        print(f"{e:>6} {dmean(e):>8.4f} | "
              f"{amean('recovery_error','RHSSA_huber',e):>8.4f} {amean('recovery_error','RLSSA_l1',e):>8.4f} | "
              f"{amean('n_iter','RHSSA_huber',e):>6.1f} {amean('n_iter','RLSSA_l1',e):>6.1f} | "
              f"{amean('secs','RHSSA_huber',e):>6.2f} {amean('secs','RLSSA_l1',e):>6.2f}")
    print(f"\n[checks] {sum(checks.values())}/{len(checks)} pass: {checks}")
    print(f"[done] wrote {out}/")


if __name__ == "__main__":
    main()
