from __future__ import annotations

import h5py
import numpy as np

__all__ = ["save_mesh", "read_mesh"]


def save_mesh(mesh: dict, path: str):

    """
    Save mesh
    """

    with h5py.File(path, "w") as f:

        for key, value in mesh.items():

            if isinstance(value, np.ndarray):

                f.create_dataset(
                    key,
                    shape=value.shape,
                    dtype=value.dtype,
                    data=value,
                    compression="gzip",
                    chunks=True,
                )

            elif isinstance(value, bytes):
                value = value.strip()

                f.attrs[key] = value

            else:

                f.attrs[key] = value


def read_mesh(path: str) -> dict:

    """
    Read mesh
    """

    mesh = {}

    with h5py.File(path, "r") as f:

        ac = "active_cell" in list(f.keys())

        for attr in list(f.attrs.keys()):

            mesh[attr] = f.attrs[attr]

        for ds in list(f.keys()):

            if ds in ["flow", "drained_area"]:

                if ac:

                    mesh[ds] = np.ma.masked_array(
                        f[ds][:], mask=(1 - f["active_cell"][:])
                    )

                else:

                    mesh[ds] = f[ds][:]

            else:

                mesh[ds] = f[ds][:]

    return mesh
