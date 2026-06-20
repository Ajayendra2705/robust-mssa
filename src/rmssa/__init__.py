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

__version__ = "0.1.0"

__all__ = [
    # embedding
    "trajectory_matrix",
    "mssa_trajectory_matrix",
    "embed",
    # decomposition
    "Decomposition",
    "DecompositionBackend",
    "StandardSVD",
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
    "__version__",
]
