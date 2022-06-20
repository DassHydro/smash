from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver.m_mesh import MeshDT

from smash.solver.m_utils import (
    sparse_matrix_to_vector_r,
    sparse_matrix_to_vector_i,
    sparse_vector_to_matrix_r,
    sparse_vector_to_matrix_i,
)

import numpy as np

def sparse_matrix_to_vector(mesh: MeshDT, matrix: np.ndarray) -> np.ndarray:

    if np.issubdtype(matrix.dtype, np.integer):

        vector = np.zeros(shape=mesh.nac, dtype=np.int32, order="F")

        sparse_matrix_to_vector_i(mesh, matrix, vector)

    else:

        vector = np.zeros(shape=mesh.nac, dtype=np.float32, order="F")

        sparse_matrix_to_vector_r(mesh, matrix, vector)

    return vector


def sparse_vector_to_matrix(mesh: MeshDT, vector: np.ndarray) -> np.ndarray:

    if np.issubdtype(vector.dtype, np.integer):

        matrix = np.zeros(shape=(mesh.nrow, mesh.ncol), dtype=np.int32, order="F")

        sparse_vector_to_matrix_i(mesh, vector, matrix)

    else:

        matrix = np.zeros(shape=(mesh.nrow, mesh.ncol), dtype=np.float32, order="F")

        sparse_vector_to_matrix_r(mesh, vector, matrix)

    matrix = np.where(matrix == -99, np.nan, matrix)

    return matrix
