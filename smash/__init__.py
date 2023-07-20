# smash main-package
from smash.core.model.model import Model

from smash.core.signal_analysis.metrics.metrics import metrics
from smash.core.signal_analysis.segmentation.segmentation import hydrograph_segmentation
from smash.core.signal_analysis.signatures.signatures import signatures

from smash.core.simulation import forward_run, optimize

# smash sub-packages
from smash import io, factory

# smash objects
from smash.core.signal_analysis.signatures.signatures import Signatures

from smash.factory.samples.samples import Samples

from . import _version


def __getattr__(name):
    raise AttributeError(f"module 'smash' has no attribute '{name}'")


__all__ = [
    "Model",
    "metrics",
    "hydrograph_segmentation",
    "signatures",
    "forward_run",
    "optimize",
    "io",
    "factory",
    "Signatures",
    "Samples",
]

__version__ = _version.get_versions()["version"]

__all__.extend(["__version__"])
