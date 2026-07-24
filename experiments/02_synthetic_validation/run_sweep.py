"""Day-17 contamination sweep on synthetic panels.

Runs the 2x2 factorial {classical, RHSSA-Huber, RLSSA-L1} x {univariate, multivariate}
over a wide contamination range epsilon in {0, 1, 5, 10, 20}% and several seeds,
scoring TWO ground-truth metrics per config from a single fit each:

  * recovery_error  -- ||recovered signal - clean S||_F / ||S||_F;
  * subspace_error  -- sin of the largest principal angle between the estimated and
    the true leading factor subspace.

Writes a tidy long CSV (one row per eps x method x mode x seed, both metrics), a
two-panel plot (each metric vs epsilon), and a JSON summary with pass/fail checks.

    python experiments/02_synthetic_validation/run_sweep.py \
        --config experiments/configs/sweep_synthetic.yaml
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
from _grid_common import MODES, make_backends, evaluate  # noqa: E402

METRICS = ["recovery_error", "subspace_error"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/sweep_synthetic.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    pan, ms, ex = cfg["panel"], cfg["mssa"], cfg["experiment"]
    L, r = ms["window"], ms["rank"]
    seeds, eps_list = ex["seeds"], ex["contamination"]
    sv = cfg.get("solver", {})
    BACKENDS = make_backends(sv.get("max_iter", 200), sv.get("tol", 1e-9))

    rows = []
    for eps in eps_list:
        for seed in seeds:
            sp = make_synthetic_panel(
                T=pan["T"], p=pan["p"], k=pan["k"], noise_sd=pan["noise_sd"],
                contamination=eps, outlier_scale=pan["outlier_scale"], seed=seed,
            )
            for method, factory in BACKENDS.items():
                for mode in MODES:
                    rec_err, sub_err = evaluate(sp.X, sp.signal, factory, L, r, mode)
                    rows.append(dict(eps=eps, method=method, mode=mode, seed=seed,
                                     recovery_error=rec_err, subspace_error=sub_err))
        print(f"  eps={eps} done")

    # ---- tidy CSV
    csv_path = out / "sweep_metrics.csv"
    header = "eps,method,mode,seed,recovery_error,subspace_error"
    lines = [header] + [
        f"{x['eps']},{x['method']},{x['mode']},{x['seed']},"
        f"{x['recovery_error']:.6f},{x['subspace_error']:.6f}" for x in rows
    ]
    csv_path.write_text("\n".join(lines) + "\n")

    def mean_metric(metric, method, mode, eps):
        vals = [x[metric] for x in rows
                if x["method"] == method and x["mode"] == mode and x["eps"] == eps]
        return float(np.mean(vals))

    # ---- two-panel plot: each metric vs contamination
    styles = {"univariate": "--", "multivariate": "-"}
    colors = {"classical": "#444444", "RHSSA_huber": "#1f77b4", "RLSSA_l1": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, METRICS):
        for method in BACKENDS:
            for mode in MODES:
                ys = [mean_metric(metric, method, mode, e) for e in eps_list]
                ax.plot([100 * e for e in eps_list], ys, styles[mode], color=colors[method],
                        marker="o", label=f"{method} · {mode}")
        ax.set_xlabel("contamination epsilon (%)")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(metric.replace("_", " "))
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"Day-17 contamination sweep (L={L}, r={r}, {len(seeds)} seeds)")
    fig.tight_layout()
    fig.savefig(out / "sweep_metrics.png", dpi=130)

    # ---- checks: for each metric, at high eps robust beats classical (both modes);
    #      at eps0 robust is close to classical (no tax).
    checks = {}
    eps0, eps_hi = min(eps_list), max(eps_list)
    for metric in METRICS:
        for mode in MODES:
            base0 = mean_metric(metric, "classical", mode, eps0)
            for method in ("RHSSA_huber", "RLSSA_l1"):
                m0 = mean_metric(metric, method, mode, eps0)
                checks[f"clean_close::{metric}::{method}::{mode}"] = bool(
                    m0 <= base0 * 1.08 + 1e-3
                )
                checks[f"robust_beats_classical::{metric}::{method}::{mode}"] = bool(
                    mean_metric(metric, method, mode, eps_hi)
                    < mean_metric(metric, "classical", mode, eps_hi)
                )

    summary = {
        "config": cfg,
        "mean": {
            metric: {
                f"{method}|{mode}": {str(e): round(mean_metric(metric, method, mode, e), 5)
                                     for e in eps_list}
                for method in BACKENDS for mode in MODES
            }
            for metric in METRICS
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    (out / "sweep_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- console report
    print(f"\n[sweep] L={L} r={r} seeds={seeds} eps={eps_list}")
    for metric in METRICS:
        print(f"\n== {metric} ==")
        print(f"{'config':28s} " + "  ".join(f"e={e:<6}" for e in eps_list))
        for method in BACKENDS:
            for mode in MODES:
                cells = "  ".join(f"{mean_metric(metric, method, mode, e):.4f} " for e in eps_list)
                print(f"{method + ' · ' + mode:28s} {cells}")
    n_pass = sum(checks.values())
    print(f"\n[checks] {n_pass}/{len(checks)} pass; all_checks_pass={summary['all_checks_pass']}")
    for k, v in checks.items():
        if not v:
            print(f"  FAIL  {k}")
    print(f"[done] wrote {out}/ (sweep_metrics.csv, sweep_metrics.png, sweep_summary.json)")


if __name__ == "__main__":
    main()
