"""Phase-1 baseline reproduction (Days 8-10): standard MSSA on an equity panel.

Reproducible entry point:

    python experiments/01_baseline_repro/run_baseline.py \
        --config experiments/configs/baseline.yaml

Loads the configured panel (Yahoo with an offline synthetic fallback), fits standard
MSSA, writes scree / w-correlation / component plots and a JSON metrics summary to the
configured output_dir. Everything is seeded; re-running reproduces identical numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from rmssa.mssa import MSSA
from rmssa.datasets import load_yahoo, make_synthetic_panel
from rmssa.plots import plot_scree, plot_wcorrelation, plot_components, savefig


def _load_panel(cfg: dict):
    """Return (X, labels, source_used). X is (T, p)."""
    d = cfg["data"]
    if d["source"] == "yahoo":
        try:
            df = load_yahoo(
                d["tickers"], start=d["start"], end=d["end"],
                field=d.get("field", "Close"), returns=d.get("returns", True),
            )
            return df.to_numpy(), list(df.columns), "yahoo"
        except RuntimeError as exc:
            print(f"[warn] Yahoo load failed ({exc}); falling back to synthetic.")
    # synthetic (either requested, or fallback)
    s = d["synthetic"]
    sp = make_synthetic_panel(
        T=s["T"], p=s["p"], k=s["k"], noise_sd=s["noise_sd"],
        contamination=s.get("contamination", 0.0), seed=cfg.get("seed", 0),
    )
    labels = [f"series_{i}" for i in range(s["p"])]
    return sp.X, labels, "synthetic"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/baseline.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    np.random.seed(cfg.get("seed", 0))
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    X, labels, source = _load_panel(cfg)
    T, p = X.shape
    print(f"[data] source={source} panel shape T={T}, p={p}, labels={labels}")

    # standardise each channel (zero mean / unit variance) before MSSA
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    channels = [Xs[:, j] for j in range(p)]

    L = cfg["mssa"]["window"]
    model = MSSA(window=L, rank=cfg["mssa"].get("rank")).fit(channels)
    contrib = model.contributions()
    print(f"[mssa] L={L}, numerical rank={model.decomposition.rank}, "
          f"top-5 contributions(%)={np.round(100*contrib[:5],3)}")

    # grouping from config
    groups = {k: v for k, v in cfg["grouping"].items()}
    comps = model.reconstruct(groups)  # {label: (p, T)}

    # ---- diagnostics & plots
    dg = cfg["diagnostics"]
    savefig(plot_scree(contrib, n=dg["scree_n"], title=f"Scree — standard MSSA (L={L})"),
            out / "scree.png")
    W = model.wcorrelation(n_components=dg["wcorr_n"])
    savefig(plot_wcorrelation(W, title="Elementary-component w-correlation"),
            out / "wcorrelation.png")
    # component plot for the first channel
    ch0 = {label: comps[label][0] for label in groups}
    savefig(plot_components(Xs[:, 0], ch0, title=f"{labels[0]} — MSSA components"),
            out / "components_channel0.png")

    # ---- reconstruction-identity check (full grouping == original)
    full = model.reconstruct_full()
    rel_err = float(np.linalg.norm(full - Xs.T) / np.linalg.norm(Xs.T))

    summary = {
        "source": source,
        "T": T, "p": p, "labels": labels,
        "window_L": L,
        "numerical_rank": int(model.decomposition.rank),
        "top10_contributions_pct": [float(x) for x in np.round(100 * contrib[:10], 4)],
        "reconstruction_rel_error": rel_err,
        "groups": {k: list(map(int, v)) for k, v in groups.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] reconstruction rel. error={rel_err:.2e}; wrote outputs to {out}/")


if __name__ == "__main__":
    main()
