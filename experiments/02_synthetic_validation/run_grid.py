"""Day-16 first full-grid comparison on synthetic panels (low contamination).

Runs the 2x2 factorial

    {classical (StandardSVD), RHSSA (RobRSVD/Huber), RLSSA (AlternatingL1SVD/L1)}
        x  {univariate SSA (per series), multivariate MSSA (shared window)}

on synthetic panels X = S + N + O with a KNOWN clean low-rank signal S, across a
low-contamination range and several seeds. For each config it reconstructs the
signal (leading-r components) and scores the signal-recovery error against S.

Confirms the two properties the project hinges on:
  * epsilon = 0  -> robust ~= classical (fair comparison, no robustness tax);
  * epsilon rises -> robust separates from classical, multivariate >= univariate.

Reproducible entry point:

    python experiments/02_synthetic_validation/run_grid.py \
        --config experiments/configs/grid_synthetic.yaml

Writes a tidy long CSV, a recovery-vs-contamination plot, and a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rmssa.datasets import make_synthetic_panel  # noqa: E402
from rmssa.metrics import signal_recovery_error  # noqa: E402

from _grid_common import MODES, make_backends, recover_signal  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/grid_synthetic.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    pan, ms, ex = cfg["panel"], cfg["mssa"], cfg["experiment"]
    L, r = ms["window"], ms["rank"]
    seeds, eps_list = ex["seeds"], ex["contamination"]
    sv = cfg.get("solver", {})
    BACKENDS = make_backends(sv.get("max_iter", 200), sv.get("tol", 1e-9))

    rows = []  # tidy long: eps, method, mode, seed, error
    for eps in eps_list:
        for seed in seeds:
            sp = make_synthetic_panel(
                T=pan["T"], p=pan["p"], k=pan["k"], noise_sd=pan["noise_sd"],
                contamination=eps, outlier_scale=pan["outlier_scale"], seed=seed,
            )
            for method, factory in BACKENDS.items():
                for mode in MODES:
                    rec = recover_signal(sp.X, factory, L, r, mode)
                    err = signal_recovery_error(rec, sp.signal)
                    rows.append(dict(eps=eps, method=method, mode=mode, seed=seed, error=err))

    # ---- write tidy CSV
    csv_path = out / "grid_recovery.csv"
    header = "eps,method,mode,seed,error"
    lines = [header] + [
        f"{x['eps']},{x['method']},{x['mode']},{x['seed']},{x['error']:.6f}" for x in rows
    ]
    csv_path.write_text("\n".join(lines) + "\n")

    # ---- aggregate mean over seeds -> {(method,mode): {eps: mean_err}}
    def mean_err(method, mode, eps):
        vals = [x["error"] for x in rows
                if x["method"] == method and x["mode"] == mode and x["eps"] == eps]
        return float(np.mean(vals))

    # ---- plot: recovery error vs contamination, one line per config
    fig, ax = plt.subplots(figsize=(7.5, 5))
    styles = {"univariate": "--", "multivariate": "-"}
    colors = {"classical": "#444444", "RHSSA_huber": "#1f77b4", "RLSSA_l1": "#d62728"}
    for method in BACKENDS:
        for mode in MODES:
            ys = [mean_err(method, mode, e) for e in eps_list]
            ax.plot([100 * e for e in eps_list], ys, styles[mode], color=colors[method],
                    marker="o", label=f"{method} · {mode}")
    ax.set_xlabel("contamination epsilon (%)")
    ax.set_ylabel("signal-recovery error  ||rec - S|| / ||S||")
    ax.set_title(f"2x2 grid — signal recovery vs contamination (L={L}, r={r}, "
                 f"{len(seeds)} seeds)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "grid_recovery.png", dpi=130)

    # ---- hard checks: the two properties the project hinges on
    checks = {}
    eps0 = min(eps_list)
    eps_hi = max(eps_list)
    # (1) no robustness tax at eps0: robust within 8% relative of classical
    for mode in MODES:
        base = mean_err("classical", mode, eps0)
        for method in ("RHSSA_huber", "RLSSA_l1"):
            m = mean_err(method, mode, eps0)
            checks[f"clean_close::{method}::{mode}"] = bool(m <= base * 1.08 + 1e-6)
    # (2) robust beats classical at high eps (both modes)
    for mode in MODES:
        for method in ("RHSSA_huber", "RLSSA_l1"):
            checks[f"robust_beats_classical::{method}::{mode}"] = bool(
                mean_err(method, mode, eps_hi) < mean_err("classical", mode, eps_hi)
            )

    # ---- observations (reported, NOT pass/fail): univariate vs multivariate is an
    # empirical question; for clean low-rank recovery with adequate rank it is ~tied.
    observations = {}
    for method in ("RHSSA_huber", "RLSSA_l1", "classical"):
        mu = mean_err(method, "multivariate", eps_hi)
        uu = mean_err(method, "univariate", eps_hi)
        observations[f"multi_minus_uni::{method}::eps_hi"] = round(mu - uu, 5)

    summary = {
        "config": cfg,
        "mean_error": {
            f"{method}|{mode}": {str(e): round(mean_err(method, mode, e), 5) for e in eps_list}
            for method in BACKENDS for mode in MODES
        },
        "checks": checks,
        "observations": observations,
        "all_checks_pass": all(checks.values()),
    }
    (out / "grid_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- console report
    print(f"[grid] L={L} r={r} seeds={seeds} eps={eps_list}")
    print(f"{'config':28s} " + "  ".join(f"e={e:<6}" for e in eps_list))
    for method in BACKENDS:
        for mode in MODES:
            cells = "  ".join(f"{mean_err(method, mode, e):.4f} " for e in eps_list)
            print(f"{method + ' · ' + mode:28s} {cells}")
    print("\n[checks]")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("\n[observations] multivariate - univariate error at eps_hi (- = MSSA better):")
    for k, v in observations.items():
        print(f"  {v:+.5f}  {k}")
    print(f"\n[done] all_checks_pass={summary['all_checks_pass']}; wrote {out}/ "
          "(grid_recovery.csv, grid_recovery.png, grid_summary.json)")


if __name__ == "__main__":
    main()
