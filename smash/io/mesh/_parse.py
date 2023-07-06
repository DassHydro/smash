from __future__ import annotations

import numpy as np


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
