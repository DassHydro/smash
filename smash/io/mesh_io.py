from __future__ import annotations

from smash.io._error import ReadHDF5MethodError

import os
import errno
import h5py
import numpy as np


__all__ = ["save_mesh", "read_mesh"]


def _parse_mesh_dict_to_hdf5(mesh: dict, hdf5_ins):
    for key, value in mesh.items():
        if isinstance(value, np.ndarray):
            if value.dtype.char == "U":
                value = value.astype("S")

            hdf5_ins.create_dataset(
                key,
                shape=value.shape,
                dtype=value.dtype,
                data=value,
                compression="gzip",
                chunks=True,
            )

        else:
            hdf5_ins.attrs[key] = value


def _parse_hdf5_to_mesh_dict(hdf5_ins) -> dict:
    mesh = {}

    attr_keys = list(hdf5_ins.attrs.keys())
    attr_keys.remove("_save_func")

    keys = list(hdf5_ins.keys())

    ac_check = "active_cell" in keys

    for key in attr_keys:
        mesh[key] = hdf5_ins.attrs[key]

    for key in keys:
        if key in ["flwdir", "flwdst", "flwacc"] and ac_check:
            mesh[key] = np.ma.masked_array(
                hdf5_ins[key][:], mask=(1 - hdf5_ins["active_cell"][:])
            )

        elif key == "code":
            mesh[key] = hdf5_ins[key][:].astype("U")

        else:
            mesh[key] = hdf5_ins[key][:]

    return mesh


def save_mesh(mesh: dict, path: str):
    """
    Save Model initialization mesh dictionary.

    Parameters
    ----------
    mesh : dict
        The mesh dictionary to be saved to `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`__ file.

    path : str
        The file path. If the path not end with ``.hdf5``, the extension is automatically added to the file path.

    See Also
    --------
    read_mesh: Read Model initialization mesh dictionary.

    Examples
    --------
    >>> setup, mesh = smash.load_dataset("cance")
    >>> mesh
    {'dx': 1000.0, 'nac': 383, 'ncol': 28, 'ng': 3, 'nrow': 28 ...}

    Save mesh

    >>> smash.save_mesh(mesh, "mesh.hdf5")

    Read mesh

    >>> mesh_rld = smash.read_mesh("mesh.hdf5")
    {'dx': 1000.0, 'nac': 383, 'ncol': 28, 'ng': 3, 'nrow': 28, ...}
    """

    if not path.endswith(".hdf5"):
        path = path + ".hdf5"

    with h5py.File(path, "w") as f:
        _parse_mesh_dict_to_hdf5(mesh, f)

        f.attrs["_save_func"] = "save_mesh"


def read_mesh(path: str) -> dict:
    """
    Read Model initialization mesh dictionary.

    Parameters
    ----------
    path : str
        The file path.

    Returns
    -------
    dict :
        A mesh dictionary loaded from HDF5 file.

    Raises
    ------
    FileNotFoundError:
        If file not found.
    ReadHDF5MethodError:
        If file not created with `save_mesh`.

    See Also
    --------
    save_mesh: Save Model initialization mesh dictionary.

    Examples
    --------
    >>> setup, mesh = smash.load_dataset("cance")
    >>> mesh
    {'dx': 1000.0, 'nac': 383, 'ncol': 28, 'ng': 3, 'nrow': 28 ...}

    Save mesh

    >>> smash.save_mesh(mesh, "mesh.hdf5")

    Read mesh

    >>> mesh_rld = smash.read_mesh("mesh.hdf5")
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
