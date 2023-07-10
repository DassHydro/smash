# smash objects
from smash.core.model import Model
from smash.signal_analysis.signatures.signatures import Signatures
from smash.factory.samples.samples import Samples

# smash sub-packages
from smash import io, factory, signal_analysis, simulation

from . import _version


def __getattr__(name):
    raise AttributeError(f"module 'smash' has no attribute '{name}'")


__all__ = [
    "Model",
    "Signatures",
    "Samples",
    "io",
    "factory",
    "signal_analysis",
    "simulation",
]

__version__ = _version.get_versions()["version"]

__all__.extend(["__version__"])
