from __future__ import annotations

import errno
import os
from typing import TYPE_CHECKING

import h5py

from smash.io._error import ReadHDF5MethodError
from smash.io.handler._hdf5_handler import _dump_dict, _load_hdf5_to_dict

if TYPE_CHECKING:
    from typing import Any

    from smash.util._typing import FilePath


__all__ = ["read_mesh", "save_mesh"]


def save_mesh(mesh: dict[str, Any], path: FilePath):
    """
    Save the Model initialization mesh dictionary to HDF5.

    Parameters
    ----------
    mesh : `dict[str, Any]`
        The mesh dictionary to be saved to `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`__ file.

    path : `str`
        The file path. If the path not end with ``.hdf5``, the extension is automatically added to the file
        path.

    See Also
    --------
    read_mesh : Read the Model initialization mesh dictionary from HDF5.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> from smash.io import save_mesh, read_mesh
    >>> setup, mesh = load_dataset("cance")
    >>> mesh
    {'nac': 383, 'ncol': 28, 'ng': 3, 'nrow': 28 ...}

    Save mesh to HDF5

    >>> save_mesh(mesh, "mesh.hdf5")
    """

    if not path.endswith(".hdf5"):
        path += ".hdf5"

    with h5py.File(path, "w") as h5:
        _dump_dict("mesh", mesh, h5)
        h5.attrs["_save_func"] = "save_mesh"


def read_mesh(path: FilePath) -> dict[str, Any]:
    """
    Read the Model initialization mesh dictionary from HDF5.

    Parameters
    ----------
    path : `str`
        The file path.

    Returns
    -------
    mesh : `dict[str, Any]`
        A mesh dictionary loaded from HDF5.

    Raises
    ------
    FileNotFoundError:
        If file not found.
    ReadHDF5MethodError:
        If file not created with `save_mesh`.

    See Also
    --------
    save_mesh : Save the Model initialization mesh dictionary to HDF5.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> from smash.io import save_mesh, read_mesh
    >>> setup, mesh = load_dataset("cance")
    >>> mesh
    {'nac': 383, 'ncol': 28, 'ng': 3, 'nrow': 28 ...}

    Save mesh to HDF5

    >>> save_mesh(mesh, "mesh.hdf5")

    Read mesh from HDF5

    >>> mesh_rld = read_mesh("mesh.hdf5")
    >>> mesh_rld
    {'nac': 383, 'ncol': 28, 'ng': 3, 'nrow': 28 ...}
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    with h5py.File(path, "r") as h5:
        if h5.attrs.get("_save_func") == "save_mesh":
            mesh = _load_hdf5_to_dict(h5["mesh"])

        else:
            raise ReadHDF5MethodError(
                f"Unable to read '{path}' with 'read_mesh' method. The file may not have been created with "
                f"'save_mesh' method."
            )

    return mesh
