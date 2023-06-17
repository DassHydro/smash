from smash.core.model import Model

from smash.mesh.mesh import generate_mesh

from smash.io.setup_io import save_setup, read_setup
from smash.io.mesh_io import save_mesh, read_mesh

from smash.dataset.load import load_dataset

from . import _version


def __getattr__(name):
    raise AttributeError(f"module 'smash' has no attribute '{name}'")


__all__ = [
    "Model",
    "generate_mesh",
    "save_setup",
    "read_setup",
    "save_mesh",
    "read_mesh",
    "load_dataset",
]

__version__ = _version.get_versions()["version"]

__all__.extend(["__version__"])
