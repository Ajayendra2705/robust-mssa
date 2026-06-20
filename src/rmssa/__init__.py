"""rmssa — Robust Multivariate Singular Spectrum Analysis.

Day-5 surface: embedding (trajectory / block-Hankel matrices) and the standard
SVD decomposition backend behind a pluggable interface. Grouping, reconstruction,
forecasting and robust backends are added in later phases (see PLAN.md).
"""

from .embedding import trajectory_matrix, mssa_trajectory_matrix, embed
from .decomposition import (
    Decomposition,
    DecompositionBackend,
    StandardSVD,
)

__version__ = "0.1.0"

__all__ = [
    "trajectory_matrix",
    "mssa_trajectory_matrix",
    "embed",
    "Decomposition",
    "DecompositionBackend",
    "StandardSVD",
    "__version__",
]
