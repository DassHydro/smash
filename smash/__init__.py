from smash import factory, io
from smash._version_meson import __version__
from smash.core.model.model import Model
from smash.core.signal_analysis.metrics.metrics import metrics
from smash.core.signal_analysis.prcp_indices.prcp_indices import PrecipitationIndices, precipitation_indices
from smash.core.signal_analysis.segmentation.segmentation import hydrograph_segmentation
from smash.core.signal_analysis.signatures.signatures import Signatures, signatures
from smash.core.simulation.control import (
    bayesian_optimize_control_info,
    optimize_control_info,
)
from smash.core.simulation.estimate.estimate import MultisetEstimate, multiset_estimate
from smash.core.simulation.optimize.optimize import (
    BayesianOptimize,
    MultipleOptimize,
    Optimize,
    bayesian_optimize,
    multiple_optimize,
    optimize,
)
from smash.core.simulation.options import (
    default_bayesian_optimize_options,
    default_optimize_options,
)
from smash.core.simulation.run.run import ForwardRun, MultipleForwardRun, forward_run, multiple_forward_run
from smash.factory.samples.samples import Samples


def __getattr__(name):
    raise AttributeError(f"module 'smash' has no attribute '{name}'")


__all__ = [
    "Model",
    "metrics",
    "hydrograph_segmentation",
    "signatures",
    "precipitation_indices",
    "forward_run",
    "multiple_forward_run",
    "optimize",
    "multiple_optimize",
    "bayesian_optimize",
    "default_optimize_options",
    "default_bayesian_optimize_options",
    "optimize_control_info",
    "bayesian_optimize_control_info",
    "multiset_estimate",
    "io",
    "factory",
    "ForwardRun",
    "MultipleForwardRun",
    "Optimize",
    "MultipleOptimize",
    "BayesianOptimize",
    "MultisetEstimate",
    "Signatures",
    "PrecipitationIndices",
    "Samples",
]

__all__ += ["__version__"]
