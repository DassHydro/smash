from smash.core.model import Model

from smash.factory.mesh import mesh
from smash.factory.dataset import dataset
from smash.factory.net import net

from smash.io.setupio import setupio
from smash.io.meshio import meshio

from smash.signal_analysis.segmentation import segmentation

from smash.signal_analysis.signatures import signatures
from smash.signal_analysis.signatures.signatures import Signatures

from . import _version


def __getattr__(name):
    raise AttributeError(f"module 'smash' has no attribute '{name}'")


__all__ = ["Model", "Signatures"]

__version__ = _version.get_versions()["version"]

__all__.extend(["__version__"])
