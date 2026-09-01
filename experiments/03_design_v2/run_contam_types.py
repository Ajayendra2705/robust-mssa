"""Design-v2 experiment 2 — outlier type x magnitude x rate, on a real base series.

Phase 2 used one contamination model (isolated additive spikes at 8x sd). The
supervisor asked for "several types of outliers and different percentages", and that
the base series be a simple real one. So this sweeps

    kind      in {additive, patch, level_shift, innovational}
    magnitude in {3, 5, 8} x sd        (8x was Phase 2's only setting, and is generous)
    eps       in {1, 5, 10, 20}%

over classical vs robust x univariate vs multivariate, with AirPassengers as the clean
signal. The question is not whether robust beats classical at 8x — Phase 2 already
showed a ~27x margin — but whether the margin survives *small* outliers and *structured*
ones. Patches and level shifts locally resemble signal, which is exactly what an
M-estimator is not built to reject.

A second, smaller study sweeps the truncation rank r. Phase 2 established the lower
bound (r below the signal's SSA-rank makes robust mistake signal for outliers). This
one establishes the upper bound, which turns out to matter just as much: a rank-r model
in an L-dimensional space with r close to L can fit the outliers exactly, and the robust
gain decays to nothing. Rank selection is therefore a first-class design factor here,
not a fixed constant.

Usage:  python run_contam_types.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import MODES, evaluate, make_backends  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rmssa.contamination import CONTAMINATION_KINDS  # noqa: E402
from rmssa.datasets import make_panel  # noqa: E402

HERE = Path(__file__).resolve().parent
REPORT = Path(__file__).resolve().parents[2] / "report" / "results_v2_contamination.md"

BASE, T, L, P, K = "airpassengers", 144, 48, 4, 2
DEPENDENCE = "shared"
# co2 ahead of sunspots: the second factor sets the panel's effective rank, and the
# smooth series keeps the clean signal genuinely low-rank (floor 0.135 vs 0.266 at r=8)
BANK_NAMES = ["airpassengers", "co2", "sunspots"]
MAGNITUDES = (3.0, 5.0, 8.0)
EPSILONS = (0.01, 0.05, 0.10, 0.20)
SEEDS = range(3)
METHODS = ("classical", "RHSSA_huber", "RLSSA_l1")

#: main grid rank — the elbow of the clean scree
RANK = 8
#: rank sub-study
RANK_SWEEP = (4, 6, 8, 12, 16, 20)
RANK_SWEEP_EPS, RANK_SWEEP_MAG = 0.05, 8.0


def panel(eps, kind, mag, seed):
    return make_panel(T=T, p=P, k=K, base=BASE, bank_method="segment",
                      bank_names=BANK_NAMES, dependence=DEPENDENCE,
                      contamination=eps, kind=kind, magnitude=mag, seed=seed)


def run_main() -> list[dict]:
    backends = make_backends()
    rows = []
    for kind in CONTAMINATION_KINDS:
        for mag in MAGNITUDES:
            for eps in EPSILONS:
                for seed in SEEDS:
                    pan = panel(eps, kind, mag, seed)
                    for method in METHODS:
                        for mode in MODES:
                            rec, sub = evaluate(pan.X, pan.signal, backends[method], L, RANK, mode)
                            rows.append({
                                "kind": kind, "magnitude": mag, "eps": eps, "seed": seed,
                                "method": method, "mode": mode, "rank": RANK,
                                "recovery": rec, "subspace": sub,
                                "realised_eps": pan.contamination_rate,
                            })
    return rows


def run_rank_sweep() -> list[dict]:
    """How the robust advantage depends on the truncation rank, per outlier type."""
    backends = make_backends()
    rows = []
    for kind in CONTAMINATION_KINDS:
        for r in RANK_SWEEP:
            for seed in SEEDS:
                clean = panel(0.0, kind, RANK_SWEEP_MAG, seed)
                floor, _ = evaluate(clean.X, clean.signal, backends["classical"], L, r, "multivariate")
                pan = panel(RANK_SWEEP_EPS, kind, RANK_SWEEP_MAG, seed)
                for method in ("classical", "RHSSA_huber"):
                    rec, _ = evaluate(pan.X, pan.signal, backends[method], L, r, "multivariate")
                    rows.append({"kind": kind, "rank": r, "seed": seed,
                                 "method": method, "recovery": rec, "floor": floor})
    return rows


def agg_main(rows):
    acc = defaultdict(list)
    for r in rows:
        acc[(r["kind"], r["magnitude"], r["eps"], r["method"], r["mode"])].append(
            (r["recovery"], r["subspace"], r["realised_eps"]))
    return {k: {"recovery": float(np.mean([x[0] for x in v])),
                "subspace": float(np.mean([x[1] for x in v])),
                "realised_eps": float(np.mean([x[2] for x in v]))}
            for k, v in acc.items()}


def agg_rank(rows):
    acc = defaultdict(list)
    for r in rows:
        acc[(r["kind"], r["rank"], r["method"])].append((r["recovery"], r["floor"]))
    return {k: {"recovery": float(np.mean([x[0] for x in v])),
                "floor": float(np.mean([x[1] for x in v]))}
            for k, v in acc.items()}


def write_report(main_rows, main, rank_rows, rank, elapsed):
    lines = [
        "# Design v2 — experiment 2: outlier type × magnitude × rate (AirPassengers)",
        "",
        f"_Generated by `experiments/03_design_v2/run_contam_types.py` in {elapsed:.0f}s; "
        f"{len(main_rows) + len(rank_rows)} fits, {len(SEEDS)} seeds per cell._",
        "",
        f"Clean signal: **{BASE}** (T={T}) combined with **co2** as the second common "
        f"factor, panel of p={P} series, {DEPENDENCE} dependence (k={K}), window L={L}, "
        f"rank r={RANK}. Both factors are genuine observed series (`segment` bank), so the "
        "clean signal keeps a real SSA rank profile.",
        "",
        "Metric: signal-recovery error ‖Ŝ − S‖_F / ‖S‖_F against the uncontaminated "
        "signal. Lower is better.",
        "",
        "## Headline — Robust MSSA (Huber·multi) vs classical MSSA, by outlier type",
        "",
        "| outlier type | magnitude | eps | classical·multi | Huber·multi | gain (×) |",
        "|-------------|-----------|-----|-----------------|-------------|----------|",
    ]
    for kind in CONTAMINATION_KINDS:
        for mag in MAGNITUDES:
            for eps in EPSILONS:
                c = main[(kind, mag, eps, "classical", "multivariate")]["recovery"]
                h = main[(kind, mag, eps, "RHSSA_huber", "multivariate")]["recovery"]
                lines.append(f"| {kind} | {mag:.0f}× | {eps:.0%} | {c:.4f} | {h:.4f} | "
                             f"**{c / h if h > 0 else float('inf'):.1f}** |")

    lines += [
        "",
        "## Rank is a design factor, not a constant",
        "",
        f"Robust MSSA vs classical MSSA at eps={RANK_SWEEP_EPS:.0%}, magnitude "
        f"{RANK_SWEEP_MAG:.0f}×, as the truncation rank varies. `floor` is the clean-data "
        "reconstruction error at that rank — the best any method could do.",
        "",
        "| outlier type | r | floor (clean) | classical | Huber | gain (×) |",
        "|-------------|---|---------------|-----------|-------|----------|",
    ]
    for kind in CONTAMINATION_KINDS:
        for r in RANK_SWEEP:
            c = rank[(kind, r, "classical")]["recovery"]
            h = rank[(kind, r, "RHSSA_huber")]["recovery"]
            f = rank[(kind, r, "classical")]["floor"]
            lines.append(f"| {kind} | {r} | {f:.4f} | {c:.4f} | {h:.4f} | "
                         f"**{c / h if h > 0 else float('inf'):.1f}** |")

    lines += ["", "## Full grid — signal-recovery error (mean over seeds)", ""]
    for kind in CONTAMINATION_KINDS:
        lines += [
            f"### {kind}", "",
            "| magnitude | eps | realised eps | classical·uni | classical·multi | "
            "Huber·uni | Huber·multi | L1·uni | L1·multi |",
            "|-----------|-----|--------------|---------------|-----------------|"
            "-----------|-------------|--------|----------|",
        ]
        for mag in MAGNITUDES:
            for eps in EPSILONS:
                cells = [main[(kind, mag, eps, m, mode)]["recovery"]
                         for m in METHODS for mode in MODES]
                re_ = main[(kind, mag, eps, "classical", "univariate")]["realised_eps"]
                lines.append(f"| {mag:.0f}× | {eps:.0%} | {re_:.1%} | "
                             + " | ".join(f"{c:.4f}" for c in cells) + " |")
        lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    t0 = time.time()
    main_rows = run_main()
    rank_rows = run_rank_sweep()
    elapsed = time.time() - t0
    m, rk = agg_main(main_rows), agg_rank(rank_rows)
    write_report(main_rows, m, rank_rows, rk, elapsed)
    (HERE / "contam_raw.json").write_text(
        json.dumps({"main": main_rows, "rank": rank_rows}, indent=1), encoding="utf-8")

    print(f"{len(main_rows) + len(rank_rows)} fits in {elapsed:.0f}s -> {REPORT}\n")
    print(f"{'type':14s} {'mag':>4s} {'eps':>5s} {'classical':>10s} {'Huber':>8s} {'gain':>7s}")
    for kind in CONTAMINATION_KINDS:
        for mag in MAGNITUDES:
            for eps in EPSILONS:
                c = m[(kind, mag, eps, "classical", "multivariate")]["recovery"]
                h = m[(kind, mag, eps, "RHSSA_huber", "multivariate")]["recovery"]
                print(f"{kind:14s} {mag:4.0f} {eps:5.0%} {c:10.4f} {h:8.4f} {c / h:7.1f}")
    print(f"\n{'type':14s} {'r':>3s} {'floor':>7s} {'classical':>10s} {'Huber':>8s} {'gain':>7s}")
    for kind in CONTAMINATION_KINDS:
        for r in RANK_SWEEP:
            c = rk[(kind, r, "classical")]["recovery"]
            h = rk[(kind, r, "RHSSA_huber")]["recovery"]
            print(f"{kind:14s} {r:3d} {rk[(kind, r, 'classical')]['floor']:7.4f} "
                  f"{c:10.4f} {h:8.4f} {c / h:7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
