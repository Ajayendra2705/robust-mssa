"""rmssa — Robust Multivariate Singular Spectrum Analysis.

Phase-1 surface (Days 1-10): the full standard (M)SSA pipeline —
embedding -> decomposition (pluggable backend) -> grouping/diagnostics ->
reconstruction — plus the MSSA orchestrator and dataset loaders. Robust
decomposition backends and forecasting arrive in later phases (see PLAN.md).
"""

from .embedding import trajectory_matrix, mssa_trajectory_matrix, embed
from .decomposition import (
    Decomposition,
    DecompositionBackend,
    StandardSVD,
    RobustSVD,
    RobRSVD,
    AlternatingL1SVD,
)
from .reconstruction import diagonal_average, reconstruct_series, reconstruct_mssa
from .grouping import (
    wcorr_weights,
    wcorrelation_matrix,
    elementary_series,
    elementary_wcorrelation,
    suggest_groups_by_contribution,
)
from .mssa import MSSA
from .metrics import (
    principal_angles,
    subspace_distance,
    grassmann_distance,
    subspace_overlap,
    rmse,
    mae,
    relative_frobenius,
    signal_recovery_error,
    factor_stability,
)

__version__ = "0.2.0"

__all__ = [
    # embedding
    "trajectory_matrix",
    "mssa_trajectory_matrix",
    "embed",
    # decomposition
    "Decomposition",
    "DecompositionBackend",
    "StandardSVD",
    "RobustSVD",
    "RobRSVD",
    "AlternatingL1SVD",
    # reconstruction
    "diagonal_average",
    "reconstruct_series",
    "reconstruct_mssa",
    # grouping / diagnostics
    "wcorr_weights",
    "wcorrelation_matrix",
    "elementary_series",
    "elementary_wcorrelation",
    "suggest_groups_by_contribution",
    # orchestrator
    "MSSA",
    # metrics
    "principal_angles",
    "subspace_distance",
    "grassmann_distance",
    "subspace_overlap",
    "rmse",
    "mae",
    "relative_frobenius",
    "signal_recovery_error",
    "factor_stability",
    "__version__",
]
