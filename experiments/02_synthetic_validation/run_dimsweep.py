"""Day-18 sweep over factor structure & dimensions.

Varies k / p / T / L one at a time around a baseline, at fixed moderate contamination,
and measures the recovery error of classical MSSA vs Robust MSSA (Huber), multivariate.
The headline output is the **robustness gain** = classical_error / robust_error per
configuration: where is Robust MSSA worth the most, and where does its edge shrink?

Rank tracks the signal SSA-rank r = 2k + 2 (factor 0 = sinusoid + linear trend -> rank 4;
each further factor -> rank 2). Distinct factor periods are used so factors never alias.

    python experiments/02_synthetic_validation/run_dimsweep.py \
        --config experiments/configs/dimsweep_synthetic.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rmssa.datasets import make_synthetic_panel  # noqa: E402
from _grid_common import make_backends, evaluate  # noqa: E402

DIMS = ["k", "p", "T", "L"]
# sweep dimension -> the key it sets in the base-params dict ("L" lives under "window")
DIM_TO_PARAM = {"k": "k", "p": "p", "T": "T", "L": "window"}


def signal_rank(k: int) -> int:
    """Signal SSA-rank for the generator: factor 0 (sin+trend)=4, each extra factor=2."""
    return 2 * k + 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/dimsweep_synthetic.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    base, sweeps = cfg["base"], cfg["sweeps"]
    seeds = cfg["seeds"]
    pool = cfg["periods_pool"]
    sv = cfg.get("solver", {})
    BACKENDS = make_backends(sv.get("max_iter", 200), sv.get("tol", 1e-9))
    methods = ["classical", "RHSSA_huber"]

    rows = []
    for dim in DIMS:
        for val in sweeps[dim]:
            params = dict(base)
            params[DIM_TO_PARAM[dim]] = val
            k = params["k"]
            r = signal_rank(k)
            L = params["window"]
            periods = pool[:k]
            for seed in seeds:
                sp = make_synthetic_panel(
                    T=params["T"], p=params["p"], k=k, noise_sd=params["noise_sd"],
                    contamination=params["contamination"], outlier_scale=params["outlier_scale"],
                    periods=periods, seed=seed,
                )
                for method in methods:
                    rec_err, sub_err = evaluate(sp.X, sp.signal, BACKENDS[method], L, r, "multivariate")
                    rows.append(dict(dim=dim, value=val, r=r, method=method, seed=seed,
                                     recovery_error=rec_err, subspace_error=sub_err))
            print(f"  {dim}={val} (r={r}) done")

    # ---- tidy CSV
    header = "dim,value,r,method,seed,recovery_error,subspace_error"
    lines = [header] + [
        f"{x['dim']},{x['value']},{x['r']},{x['method']},{x['seed']},"
        f"{x['recovery_error']:.6f},{x['subspace_error']:.6f}" for x in rows
    ]
    (out / "dimsweep_metrics.csv").write_text("\n".join(lines) + "\n")

    def mean_err(dim, val, method):
        vals = [x["recovery_error"] for x in rows
                if x["dim"] == dim and x["value"] == val and x["method"] == method]
        return float(np.mean(vals))

    # ---- gain table: classical / robust recovery-error ratio per config
    gains = {}
    for dim in DIMS:
        gains[dim] = {}
        for val in sweeps[dim]:
            c = mean_err(dim, val, "classical")
            rob = mean_err(dim, val, "RHSSA_huber")
            gains[dim][val] = {
                "classical": round(c, 4),
                "robust": round(rob, 4),
                "gain_ratio": round(c / rob, 2) if rob > 0 else float("inf"),
            }

    # ---- 4-panel plot: recovery error vs dimension value (classical vs robust)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, dim in zip(axes.ravel(), DIMS):
        xs = sweeps[dim]
        ax.plot(xs, [mean_err(dim, v, "classical") for v in xs], "o-",
                color="#444444", label="classical MSSA")
        ax.plot(xs, [mean_err(dim, v, "RHSSA_huber") for v in xs], "s-",
                color="#1f77b4", label="Robust MSSA (Huber)")
        ax.set_xlabel(dim)
        ax.set_ylabel("recovery error")
        ax.set_yscale("log")
        ax.set_title(f"vary {dim}")
        ax.grid(True, alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"Day-18 dimension sweep (eps={base['contamination']}, {len(seeds)} seeds) "
                 "— recovery error, log scale")
    fig.tight_layout()
    fig.savefig(out / "dimsweep_recovery.png", dpi=130)

    summary = {"config": cfg, "gains": gains}
    (out / "dimsweep_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- console report
    print(f"\n[dimsweep] eps={base['contamination']} seeds={seeds}")
    print("robustness gain = classical_error / robust_error (higher = robust worth more)")
    for dim in DIMS:
        print(f"\n== vary {dim} (others at baseline) ==")
        print(f"{'value':>6}  {'r':>3}  {'classical':>10}  {'robust':>10}  {'gain':>6}")
        for val in sweeps[dim]:
            g = gains[dim][val]
            rr = next(x["r"] for x in rows if x["dim"] == dim and x["value"] == val)
            print(f"{val:>6}  {rr:>3}  {g['classical']:>10.4f}  {g['robust']:>10.4f}  {g['gain_ratio']:>6.1f}x")
    print(f"\n[done] wrote {out}/ (dimsweep_metrics.csv, dimsweep_recovery.png, dimsweep_summary.json)")


if __name__ == "__main__":
    main()
