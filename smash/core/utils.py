from __future__ import annotations

from smash.solver._mw_sparse_storage import (
    sparse_matrix_to_vector_r,
    sparse_matrix_to_vector_i,
    sparse_vector_to_matrix_r,
    sparse_vector_to_matrix_i,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver._mwd_mesh import MeshDT

import numpy as np

__all__ = ["sparse_matrix_to_vector", "sparse_vector_to_matrix"]


# % Not usefull for user might remove from public method
def sparse_matrix_to_vector(mesh: MeshDT, matrix: np.ndarray) -> np.ndarray:
    """
    Convert a NumPy 2D array to a 1D array respecting the order of the sparse storage.

    .. note::

        To avoid a memory overflow, the atmospheric forcings and simulated discharges can be sparse stored
        by precising it in the Model initialization setup dictionary ``(sparse_storage = True)``.
        It allows to store for each time step a 1D array whose size is the number of active cells instead of storing
        the whole rectangular domain. ``O(nrow * ncol)`` -> ``O(nac)`` with ``nac < nrow * ncol``.

    Parameters
    ----------
    mesh : MeshDT, the Model mesh attributes (see `Model.mesh`).

    matrix : NumPy 2D array.
        The 2D array to be converted

    Returns
    -------
    vector : NumPy 1D array.
        The 1D array respecting the order of the sparse storage.

    Examples
    --------
    >>> setup, mesh = smash.load_dataset("cance")
    >>> model = smash.Model(setup, mesh)
    """

    if np.issubdtype(matrix.dtype, np.integer):
        vector = np.zeros(shape=mesh.nac, dtype=np.int32, order="F")

        sparse_matrix_to_vector_i(mesh, matrix, vector)

    else:
        vector = np.zeros(shape=mesh.nac, dtype=np.float32, order="F")

        sparse_matrix_to_vector_r(mesh, matrix, vector)

    return vector


def sparse_vector_to_matrix(mesh: MeshDT, vector: np.ndarray) -> np.ndarray:
    """
    Convert a NumPy 1D array respecting the order of the sparse storage to a 2D array.

    .. note::

        To avoid a memory overflow, the atmospheric forcings and simulated discharges can be sparse stored
        by precising it in the Model initialization setup dictionary ``(sparse_storage = True)``.
        It allows to store for each time step a 1D array whose size is the number of active cells instead of storing
        the whole rectangular domain. ``O(nrow * ncol)`` -> ``O(nac)`` with ``nac < nrow * ncol``.

    Parameters
    ----------
    mesh : MeshDT, the Model mesh attributes (see `Model.mesh`).

    vector : NumPy 1D array.
        The 1D array respecting the order of the sparse storage of shape ``(nac)``.

    Returns
    -------
    matrix : NumPy 2D array.
        The 2D array of shape ``(nrow, ncol)``. Non active cells are filled in with NaN.

    Examples
    --------
    >>> setup, mesh = smash.load_dataset("cance")

    Precise the sparse storage in the setup.

    >>> setup["sparse_storage"] = True

    >>> model = smash.Model(setup, mesh)

    Access to the precipitation at time step 0 with sparse storing.

    >>> sparse_prcp = model.input_data.sparse_prcp[:,0]
    >>> sparse_prcp.size, sparse_prcp.shape, model.mesh.nac
    (383, (383,), 383)

    Convert to a 2D array.

    >>> prcp = smash.sparse_vector_to_matrix(model.mesh, sparse_prcp)
    >>> prcp.size, prcp.shape, model.mesh.nrow * model.mesh.ncol
    (784, (28, 28), 784)
    """

    if np.issubdtype(vector.dtype, np.integer):
        matrix = np.zeros(shape=(mesh.nrow, mesh.ncol), dtype=np.int32, order="F")

        sparse_vector_to_matrix_i(mesh, vector, matrix)

    else:
        matrix = np.zeros(shape=(mesh.nrow, mesh.ncol), dtype=np.float32, order="F")

        sparse_vector_to_matrix_r(mesh, vector, matrix)

    matrix = np.where(matrix == -99, np.nan, matrix)

    return matrix
