from __future__ import annotations

import os
import h5py
import numpy as np


__all__ = ["save_mesh", "read_mesh"]


def _parse_mesh_dict_to_hdf5(mesh: dict, hdf5_ins):

    for key, value in mesh.items():

        if isinstance(value, np.ndarray):

            hdf5_ins.create_dataset(
                key,
                shape=value.shape,
                dtype=value.dtype,
                data=value,
                compression="gzip",
                chunks=True,
            )

        elif isinstance(value, bytes):

            value = value.strip()

            hdf5_ins.attrs[key] = value

        else:

            hdf5_ins.attrs[key] = value


def _parse_hdf5_to_mesh_dict(hdf5_ins) -> dict:

    mesh = {}

    ac_check = "active_cell" in list(hdf5_ins.keys())

    for attr in hdf5_ins.attrs.keys():

        mesh[attr] = hdf5_ins.attrs[attr]

    for ds in hdf5_ins.keys():

        if ds in ["flwdir", "flwdst", "drained_area"] and ac_check:

            mesh[ds] = np.ma.masked_array(
                hdf5_ins[ds][:], mask=(1 - hdf5_ins["active_cell"][:])
            )

        else:

            mesh[ds] = hdf5_ins[ds][:]

    return mesh


def save_mesh(mesh: dict, path: str):

    """
    Save mesh
    """

    if not path.endswith(".hdf5"):

        path = path + ".hdf5"

    with h5py.File(path, "w") as f:

        _parse_mesh_dict_to_hdf5(mesh, f)

        f.close()


def read_mesh(path: str) -> dict:

    """
    Read mesh
    """

    if os.path.isfile(path):

        with h5py.File(path, "r") as f:

            mesh = _parse_hdf5_to_mesh_dict(f)

            f.close()

        return mesh

    else:

        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
