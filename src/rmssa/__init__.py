"""rmssa — Robust Multivariate Singular Spectrum Analysis.

The full (M)SSA pipeline — embedding -> decomposition (pluggable backend) ->
grouping/diagnostics -> reconstruction -> recurrent forecasting — plus the MSSA
orchestrator, dataset/panel generators and ground-truth metrics.

Backends carry the classical/robust axis (StandardSVD vs RobRSVD/AlternatingL1SVD);
MSSA.fit carries the univariate/multivariate axis. Everything downstream, forecasting
included, is written against the shared interfaces, so the 2x2 comparison needs no
special-casing anywhere.
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
from .contamination import CONTAMINATION_KINDS, Contamination, contaminate
from .datasets import (
    BASE_SERIES,
    DEPENDENCE_LEVELS,
    Panel,
    SyntheticPanel,
    factor_bank,
    load_base_series,
    make_panel,
    make_synthetic_panel,
)
from .forecasting import (
    RollingForecastResult,
    forecast,
    forecast_recurrent,
    is_explosive,
    lrr_roots,
    max_root_modulus,
    recurrent_coefficients,
    rolling_origin_forecast,
    verticality,
)

__version__ = "0.3.0"

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
    # contamination models
    "CONTAMINATION_KINDS",
    "Contamination",
    "contaminate",
    # datasets / panel generators
    "BASE_SERIES",
    "DEPENDENCE_LEVELS",
    "Panel",
    "SyntheticPanel",
    "factor_bank",
    "load_base_series",
    "make_panel",
    "make_synthetic_panel",
    # forecasting
    "RollingForecastResult",
    "forecast",
    "forecast_recurrent",
    "is_explosive",
    "lrr_roots",
    "max_root_modulus",
    "recurrent_coefficients",
    "rolling_origin_forecast",
    "verticality",
    "__version__",
]
