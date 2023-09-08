from __future__ import annotations

from smash.io._error import ReadHDF5MethodError

from smash.io.mesh._parse import _parse_hdf5_to_mesh_dict, _parse_mesh_dict_to_hdf5

import os
import errno
import h5py


__all__ = ["save_mesh", "read_mesh"]


def save_mesh(mesh: dict, path: str):
    """
    Save the Model initialization mesh dictionary.

    Parameters
    ----------
    mesh : dict
        The mesh dictionary to be saved to `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`__ file.

    path : str
        The file path. If the path not end with ``.hdf5``, the extension is automatically added to the file path.

    See Also
    --------
    read_mesh : Read the Model initialization mesh dictionary.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> from smash.io import save_mesh, read_mesh
    >>> setup, mesh = load_dataset("cance")
    >>> mesh
    {'dx': 1000.0, 'nac': 383, 'ncol': 28, 'ng': 3, 'nrow': 28 ...}

    Save mesh:

    >>> save_mesh(mesh, "mesh.hdf5")

    Read mesh:

    >>> mesh_rld = read_mesh("mesh.hdf5")
    >>> mesh_rld
    {'dx': 1000.0, 'nac': 383, 'ncol': 28, 'ng': 3, 'nrow': 28, ...}
    """

    if not path.endswith(".hdf5"):
        path = path + ".hdf5"

    with h5py.File(path, "w") as f:
        _parse_mesh_dict_to_hdf5(mesh, f)

        f.attrs["_save_func"] = "save_mesh"


def read_mesh(path: str) -> dict:
    """
    Read the Model initialization mesh dictionary.

    Parameters
    ----------
    path : str
        The file path.

    Returns
    -------
    mesh : dict
        A mesh dictionary loaded from HDF5 file.

    Raises
    ------
    FileNotFoundError:
        If file not found.
    ReadHDF5MethodError:
        If file not created with `save_mesh`.

    See Also
    --------
    save_mesh : Save the Model initialization mesh dictionary.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> from smash.io import save_mesh, read_mesh
    >>> setup, mesh = load_dataset("cance")
    >>> mesh
    {'dx': 1000.0, 'nac': 383, 'ncol': 28, 'ng': 3, 'nrow': 28 ...}

    Save mesh:

    >>> save_mesh(mesh, "mesh.hdf5")

    Read mesh:

    >>> mesh_rld = read_mesh("mesh.hdf5")
    >>> mesh_rld
    {'dx': 1000.0, 'nac': 383, 'ncol': 28, 'ng': 3, 'nrow': 28, ...}
    """

    if os.path.isfile(path):
        with h5py.File(path, "r") as f:
            if f.attrs.get("_save_func") == "save_mesh":
                return _parse_hdf5_to_mesh_dict(f)

            else:
                raise ReadHDF5MethodError(
                    f"Unable to read '{path}' with 'read_mesh' method. The file may not have been created with 'save_mesh' method."
                )

    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
