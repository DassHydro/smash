from smash.core.model import Model

from smash.factory.mesh import mesh
from smash.factory.dataset import dataset

from smash.io import setup_io
from smash.io import mesh_io

from . import _version


def __getattr__(name):
    raise AttributeError(f"module 'smash' has no attribute '{name}'")


__all__ = ["Model"]

__version__ = _version.get_versions()["version"]

__all__.extend(["__version__"])
