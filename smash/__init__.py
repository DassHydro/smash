from smash import factory, io
from smash._version_meson import __version__
from smash.core.model.model import Model
from smash.core.signal_analysis.evaluation.evaluation import evaluation
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
    Optimize,
    bayesian_optimize,
    optimize,
)
from smash.core.simulation.options import (
    default_bayesian_optimize_options,
    default_optimize_options,
)
from smash.core.simulation.run.run import (
    ForwardRun,
    MultipleForwardRun,
    forward_run,
    multiple_forward_run,
)
from smash.factory.samples.samples import Samples


def __getattr__(name):
    raise AttributeError(f"module 'smash' has no attribute '{name}'")


__all__ = [
    "BayesianOptimize",
    "ForwardRun",
    "Model",
    "MultipleForwardRun",
    "MultisetEstimate",
    "Optimize",
    "PrecipitationIndices",
    "Samples",
    "Signatures",
    "bayesian_optimize",
    "bayesian_optimize_control_info",
    "default_bayesian_optimize_options",
    "default_optimize_options",
    "evaluation",
    "factory",
    "forward_run",
    "hydrograph_segmentation",
    "io",
    "multiple_forward_run",
    "multiset_estimate",
    "optimize",
    "optimize_control_info",
    "precipitation_indices",
    "signatures",
]

__all__ += ["__version__"]
