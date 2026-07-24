"""Day-14 R cross-check driver.

Generates a fixture trajectory matrix H, saves it, runs our robust backends, then:
  * if ``Rscript`` is on PATH, runs robust_svd_reference.R and reports the subspace
    distance between our leading factor subspace and each R reference (RLSSA / RHSSA);
  * otherwise, leaves H.csv + our vectors in place and prints the one command to run
    the authoritative check later. R absence is reported as SKIPPED, not PASS/FAIL.

Usage:
    python run_rcheck.py            # rank r = 2 by default
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rmssa.datasets import make_synthetic_panel  # noqa: E402
from rmssa.embedding import mssa_trajectory_matrix  # noqa: E402
from rmssa.decomposition import RobRSVD, AlternatingL1SVD  # noqa: E402
from rmssa.metrics import subspace_distance  # noqa: E402

HERE = Path(__file__).resolve().parent
R = 2


def build_fixture(seed: int = 7):
    sp = make_synthetic_panel(T=200, p=5, k=2, noise_sd=0.03,
                              contamination=0.05, outlier_scale=8.0, seed=seed)
    channels = [sp.X[:, j] for j in range(sp.X.shape[1])]
    H, _ = mssa_trajectory_matrix(channels, 40)
    return H


def main() -> int:
    H = build_fixture()
    np.savetxt(HERE / "H.csv", H, delimiter=",")

    U_huber = RobRSVD(rank=R).decompose(H).U[:, :R]
    U_l1 = AlternatingL1SVD(rank=R).decompose(H).U[:, :R]
    np.savetxt(HERE / "py_U_huber.csv", U_huber, delimiter=",")
    np.savetxt(HERE / "py_U_l1.csv", U_l1, delimiter=",")
    print(f"fixture H {H.shape} + Python robust vectors written to {HERE}")

    rscript = shutil.which("Rscript")
    if rscript is None:
        print("\n[SKIPPED] Rscript not found on PATH.")
        print("Authoritative package-level check pending an R install. To run it:")
        print(f"    cd {HERE}")
        print("    Rscript robust_svd_reference.R . 2")
        print("    python run_rcheck.py   # re-run: it will pick up the R_*.csv outputs")
        # if a previous R run left outputs, still compare them
        if not (HERE / "R_U_l1.csv").exists():
            return 0

    else:
        print(f"\nRunning {rscript} robust_svd_reference.R ...")
        proc = subprocess.run(
            [rscript, str(HERE / "robust_svd_reference.R"), str(HERE), str(R)],
            capture_output=True, text=True,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print("[R ERROR]\n", proc.stderr)
            return 1

    ok = True
    for name, py_U, r_file in [
        ("RLSSA (L1)", U_l1, "R_U_l1.csv"),
        ("RHSSA (Huber)", U_huber, "R_U_huber.csv"),
    ]:
        r_path = HERE / r_file
        if not r_path.exists():
            print(f"  {name}: R output {r_file} missing, skipped")
            continue
        R_U = np.atleast_2d(np.loadtxt(r_path, delimiter=","))
        if R_U.shape[0] == 1:
            R_U = R_U.T
        d = subspace_distance(py_U, R_U[:, :R])
        verdict = "OK" if d < 0.1 else "DIVERGED"
        ok = ok and d < 0.1
        print(f"  {name}: subspace_distance(python, R) = {d:.4f}  [{verdict}]")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
