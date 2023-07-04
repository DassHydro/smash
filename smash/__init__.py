from smash.core.model import Model

from smash.factory.mesh import mesh
from smash.factory.dataset import dataset

from smash.io.setup_io import save_setup, read_setup
from smash.io.mesh_io import save_mesh, read_mesh

from . import _version


def __getattr__(name):
    raise AttributeError(f"module 'smash' has no attribute '{name}'")


__all__ = [
    "Model",
    "save_setup",
    "read_setup",
    "save_mesh",
    "read_mesh",
]

__version__ = _version.get_versions()["version"]

__all__.extend(["__version__"])
