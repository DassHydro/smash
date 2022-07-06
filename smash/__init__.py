from smash.core.model import Model
from smash.core.utils import sparse_matrix_to_vector, sparse_vector_to_matrix

from smash.mesh.meshing import generate_mesh

from smash.io.yaml import save_setup, read_setup
from smash.io.hdf5 import save_mesh, read_mesh

from smash.dataset.load import load_dataset


def __getattr__(name):

    import warnings

    raise AttributeError(f"module 'smash' has no attribute '{name}'")


__all__ = [
    "Model",
    "sparse_matrix_to_vector",
    "sparse_vector_to_matrix",
    "generate_mesh",
    "save_setup",
    "read_setup",
    "save_mesh",
    "read_mesh",
    "load_dataset",
]
