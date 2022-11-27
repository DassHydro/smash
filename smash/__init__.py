from smash.core.model import Model
from smash.core.generate_samples import generate_samples
from smash.core.utils import sparse_matrix_to_vector, sparse_vector_to_matrix

from smash.mesh.meshing import generate_mesh

from smash.io.setup_io import save_setup, read_setup
from smash.io.mesh_io import save_mesh, read_mesh
from smash.io.model_io import save_model, read_model

from smash.dataset.load import load_dataset


def __getattr__(name):

    import warnings

    raise AttributeError(f"module 'smash' has no attribute '{name}'")


__all__ = [
    "Model",
    "generate_samples",
    "sparse_matrix_to_vector",
    "sparse_vector_to_matrix",
    "generate_mesh",
    "save_setup",
    "read_setup",
    "save_mesh",
    "read_mesh",
    "save_model",
    "read_model",
    "load_dataset",
]

from . import _version
__version__ = _version.get_versions()['version']
