from smash.core.model import Model
from smash.core.net import Net
from smash.core.signatures import SignResult, SignSensResult
from smash.core.prcp_indices import PrcpIndicesResult
from smash.core.optimize.bayes_optimize import BayesResult
from smash.core.generate_samples import generate_samples, SampleResult
from smash.core.utils import sparse_matrix_to_vector, sparse_vector_to_matrix

from smash.mesh.meshing import generate_mesh

from smash.io.setup_io import save_setup, read_setup
from smash.io.mesh_io import save_mesh, read_mesh
from smash.io.model_io import save_model, read_model
from smash.io.model_ddt_io import save_model_ddt, read_model_ddt
from smash.io.multi_model_io import save_smash_model_to_hdf5, load_hdf5_file

from smash.dataset.load import load_dataset

from . import _version


def __getattr__(name):
    raise AttributeError(f"module 'smash' has no attribute '{name}'")


__all__ = [
    "Model",
    "Net",
    "BayesResult",
    "SignResult",
    "SignSensResult",
    "PrcpIndicesResult",
    "SampleResult",
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
    "save_model_ddt",
    "read_model_ddt",
    "save_smash_model_to_hdf5",
    "load_hdf5_file",
    "load_dataset",
]

__version__ = _version.get_versions()["version"]

__all__.extend(["__version__"])
