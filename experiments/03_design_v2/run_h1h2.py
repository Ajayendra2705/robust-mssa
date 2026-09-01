"""Design-v2 experiment 1 — the dependence axis, and the supervisor's two hypotheses.

Prof. Rodrigues (25 Jul 2026) stated two expectations that the Phase-2 design could not
test, because that generator was always fully shared:

    H1  robust SSA ~= robust MSSA when the series are *independent*;
    H2  all four combinations coincide when there is *no contamination* AND the
        variables are independent.

Both are checked here as explicit PASS/FAIL assertions over the dependence axis
{independent, partial, shared} x contamination {0, 5%} x method x mode.

The rank convention is carried as a factor rather than fixed, because it decides H1:

  * ``matched``  — both modes get the univariate rank (the naive reading);
  * ``capacity`` — each mode gets the rank the clean signal actually occupies in it,
    which under independence is several times larger for MSSA (p independent signals
    share one L-dimensional row space).

Usage:  python run_h1h2.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import MODES, clean_ranks, evaluate, make_backends  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rmssa.datasets import make_panel  # noqa: E402

HERE = Path(__file__).resolve().parent
REPORT = Path(__file__).resolve().parents[2] / "report" / "results_v2_h1h2.md"

# base -> (T, L, bank_method, bank_names, fixed_rank)
#
# The real base uses `segment` (distinct real series / windows), NOT `surrogate`.
# Surrogates are the only construction that makes real-series factors independent, but
# they destroy the SSA rank profile that makes the problem an SSA problem at all
# (AirPassengers: r=10 for 99% of trajectory variance; its surrogate: r=28). So the
# dependence axis is carried by the SYNTHETIC base, where independence is exact
# (mean|corr| ~ 0.03) and the rank is exact; the real base runs the same grid as a
# sensitivity check, with the caveat that its `independent` level is independent only
# in the loading structure — the underlying real series still co-trend, which the
# reported mean|corr| column makes visible.
#
# `fixed_rank` is None where the oracle rule is well posed (synthetic) and an explicit
# number where it is not (real). r=8 is the elbow of the clean AirPassengers+co2 scree;
# experiment 2 sweeps r and shows what the choice costs.
CONFIGS = {
    "synthetic": (300, 60, "surrogate", None, None),
    "airpassengers": (144, 48, "segment", ["airpassengers", "co2", "sunspots"], 8),
}
DEPENDENCE = ("independent", "partial", "shared")
EPSILONS = (0.0, 0.05)
SEEDS = range(5)
P, K = 4, 2

#: H1/H2 tolerances. Two configurations "coincide" if they are within TOL_REL of each
#: other *relatively* OR within TOL_ABS *absolutely*.
#:
#: The absolute clause is not a loophole, it is necessary. At eps=0 the recovery errors
#: are ~0.008 and ~0.015 — both essentially exact reconstructions — yet they differ by
#: 63% in relative terms. Judging "do these coincide?" on a ratio of two near-zero
#: numbers answers a question nobody asked. TOL_ABS = 0.02 means "both recover the
#: signal to within 2% of its norm", against contaminated errors of 0.5-1.8.
TOL_REL = 0.25
TOL_ABS = 0.02


def relgap(a: float, b: float) -> float:
    """Symmetric relative gap |a-b| / mean(a,b); 0 means the two coincide."""
    m = 0.5 * (abs(a) + abs(b))
    return abs(a - b) / m if m > 0 else 0.0


def compute_ranks() -> dict:
    """(base, dependence) -> (r_univariate, r_multivariate), from the clean signal."""
    ranks = {}
    for base, (T, L, bank, names, fixed_rank) in CONFIGS.items():
        for dep in DEPENDENCE:
            probe = make_panel(T=T, p=P, k=K, base=base, bank_method=bank,
                               bank_names=names, dependence=dep,
                               contamination=0.0, seed=0)
            ranks[(base, dep)] = ((fixed_rank, fixed_rank) if fixed_rank is not None
                                  else clean_ranks(probe.signal, L))
    return ranks


def run() -> tuple[list[dict], dict]:
    backends = make_backends()
    rows: list[dict] = []
    ranks_used: dict = {}

    for base, (T, L, bank, names, fixed_rank) in CONFIGS.items():
        for dep in DEPENDENCE:
            # ranks are read off the CLEAN signal once per (base, dependence)
            probe = make_panel(T=T, p=P, k=K, base=base, bank_method=bank,
                               bank_names=names, dependence=dep,
                               contamination=0.0, seed=0)
            if fixed_rank is None:
                r_uni, r_multi = clean_ranks(probe.signal, L)
            else:
                r_uni = r_multi = fixed_rank
            ranks_used[(base, dep)] = (r_uni, r_multi)

            for eps in EPSILONS:
                for seed in SEEDS:
                    pan = make_panel(T=T, p=P, k=K, base=base, bank_method=bank,
                                     bank_names=names, dependence=dep, contamination=eps,
                                     kind="additive", magnitude=8.0, seed=seed)
                    for method, factory in backends.items():
                        for mode in MODES:
                            for convention in ("matched", "capacity"):
                                if mode == "univariate":
                                    r = r_uni
                                else:
                                    r = r_uni if convention == "matched" else r_multi
                                rec, sub = evaluate(pan.X, pan.signal, factory, L, r, mode)
                                rows.append({
                                    "base": base, "dependence": dep, "eps": eps,
                                    "method": method, "mode": mode,
                                    "convention": convention, "rank": r, "seed": seed,
                                    "recovery": rec, "subspace": sub,
                                    "mean_abs_corr": pan.mean_abs_corr,
                                    "max_abs_corr": pan.max_abs_corr,
                                })
    return rows, ranks_used


def aggregate(rows: list[dict]) -> dict:
    """(base, dep, eps, method, mode, convention) -> mean over seeds."""
    acc = defaultdict(list)
    for r in rows:
        key = (r["base"], r["dependence"], r["eps"], r["method"], r["mode"], r["convention"])
        acc[key].append((r["recovery"], r["subspace"], r["mean_abs_corr"]))
    return {
        k: {"recovery": float(np.mean([v[0] for v in vals])),
            "subspace": float(np.mean([v[1] for v in vals])),
            "mean_abs_corr": float(np.mean([v[2] for v in vals]))}
        for k, vals in acc.items()
    }


def _verdict(a: float, b: float) -> dict:
    rel, absolute = relgap(a, b), abs(a - b)
    return {
        "a": f"{a:.4f}", "b": f"{b:.4f}",
        "rel": f"{rel:.3f}", "abs": f"{absolute:.4f}",
        "verdict": "PASS" if (rel <= TOL_REL or absolute <= TOL_ABS) else "FAIL",
    }


def check_hypotheses(agg: dict) -> list[dict]:
    """H1 and H2 as explicit checks, one row per (base, convention)."""
    checks = []
    for base in CONFIGS:
        for convention in ("matched", "capacity"):
            # ---- H1: robust SSA ~= robust MSSA under independence, WITH contamination
            uni = agg[(base, "independent", 0.05, "RHSSA_huber", "univariate", convention)]["recovery"]
            mul = agg[(base, "independent", 0.05, "RHSSA_huber", "multivariate", convention)]["recovery"]
            checks.append({
                "hypothesis": "H1", "base": base, "convention": convention,
                "detail": "robust SSA vs robust MSSA | independent, eps=5%",
                **_verdict(uni, mul),
            })

            # ---- H2: all four combinations coincide at eps=0 under independence
            vals = [agg[(base, "independent", 0.0, m, mode, convention)]["recovery"]
                    for m in ("classical", "RHSSA_huber") for mode in MODES]
            checks.append({
                "hypothesis": "H2", "base": base, "convention": convention,
                "detail": "all 4 combinations | independent, eps=0",
                **_verdict(min(vals), max(vals)),
            })
    return checks


def write_report(rows, agg, ranks_used, checks, elapsed):
    lines = [
        "# Design v2 — experiment 1: dependence axis and hypotheses H1 / H2",
        "",
        f"_Generated by `experiments/03_design_v2/run_h1h2.py` in {elapsed:.0f}s; "
        f"{len(rows)} fits, {len(SEEDS)} seeds per cell._",
        "",
        "## What this tests",
        "",
        "The supervisor's two stated expectations (25 Jul 2026):",
        "",
        "* **H1** — robust SSA ≈ robust MSSA when the series are *independent*.",
        "* **H2** — all four combinations coincide with *no contamination* and independent series.",
        "",
        "Neither was testable under the Phase-2 generator, which was always fully shared.",
        "The `dependence` factor {independent, partial, shared} supplies the missing axis.",
        "",
        "## Ranks used (read off the clean signal, 99.9% variance share)",
        "",
        "| base | dependence | r (univariate) | r (multivariate) |",
        "|------|-----------|----------------|------------------|",
    ]
    for (base, dep), (ru, rm) in ranks_used.items():
        lines.append(f"| {base} | {dep} | {ru} | {rm} |")
    lines += [
        "",
        "The gap between the two columns under `independent` is the structural point: "
        "`p` independent signals need several times more dimensions jointly than any one "
        "of them needs alone, but horizontal MSSA has a single `L`-dimensional row space "
        "to hold them all.",
        "",
        "## Hypothesis checks",
        "",
        "| hypothesis | base | rank convention | comparison | A | B | rel. gap | abs. gap | verdict |",
        "|-----------|------|-----------------|-----------|---|---|----------|----------|---------|",
    ]
    for c in checks:
        lines.append(
            f"| {c['hypothesis']} | {c['base']} | {c['convention']} | {c['detail']} | "
            f"{c['a']} | {c['b']} | {c['rel']} | {c['abs']} | **{c['verdict']}** |"
        )
    lines += [
        "",
        f"Two configurations count as coinciding if they are within {TOL_REL:.0%} of each "
        f"other relatively **or** within {TOL_ABS} absolutely. The absolute clause matters: "
        "at eps=0 the errors are ~0.008 vs ~0.015, both essentially exact reconstructions, "
        "but a ratio test on two near-zero numbers reports a 63% disagreement.",
        "",
        "## Signal-recovery error by cell (mean over seeds)",
        "",
    ]

    for base in CONFIGS:
        for convention in ("matched", "capacity"):
            lines += [
                f"### {base} — `{convention}` rank convention",
                "",
                "| dependence | mean&#124;corr&#124; | eps | classical·uni | classical·multi | "
                "Huber·uni | Huber·multi | L1·uni | L1·multi |",
                "|-----------|------------|-----|---------------|-----------------|"
                "-----------|-------------|--------|----------|",
            ]
            for dep in DEPENDENCE:
                for eps in EPSILONS:
                    cells = []
                    for method in ("classical", "RHSSA_huber", "RLSSA_l1"):
                        for mode in MODES:
                            cells.append(agg[(base, dep, eps, method, mode, convention)]["recovery"])
                    mac = agg[(base, dep, eps, "classical", "univariate", convention)]["mean_abs_corr"]
                    lines.append(
                        f"| {dep} | {mac:.3f} | {eps:.0%} | " + " | ".join(f"{c:.4f}" for c in cells) + " |"
                    )
            lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")


CACHE = HERE / "h1h2_raw.json"


def main() -> int:
    """``--from-cache`` re-scores the saved fits instead of refitting (the grid takes
    ~9 minutes; changing a tolerance or a table should not cost that)."""
    use_cache = "--from-cache" in sys.argv
    t0 = time.time()
    if use_cache:
        if not CACHE.exists():
            print(f"no cache at {CACHE}; run without --from-cache first")
            return 1
        payload = json.loads(CACHE.read_text(encoding="utf-8"))
        rows = payload["rows"] if isinstance(payload, dict) else payload
        ranks_used = compute_ranks()  # cheap: clean-signal SVDs only, no robust fits
        elapsed = 0.0
    else:
        rows, ranks_used = run()
        elapsed = time.time() - t0
        CACHE.write_text(json.dumps({"rows": rows}, indent=1), encoding="utf-8")

    agg = aggregate(rows)
    checks = check_hypotheses(agg)
    write_report(rows, agg, ranks_used, checks, elapsed)

    print(f"{len(rows)} fits {'(from cache)' if use_cache else f'in {elapsed:.0f}s'} "
          f"-> {REPORT}\n")
    for c in checks:
        print(f"  [{c['verdict']:4s}] {c['hypothesis']} {c['base']:14s} {c['convention']:9s} "
              f"A={c['a']} B={c['b']} rel={c['rel']} abs={c['abs']}")
    n_fail = sum(c["verdict"] == "FAIL" for c in checks)
    print(f"\n{len(checks) - n_fail}/{len(checks)} hypothesis checks PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
