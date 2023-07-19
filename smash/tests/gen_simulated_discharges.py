from __future__ import annotations

import smash

import os
import h5py
import numpy as np


def dump_to_baseline(f: h5py.File, key: str, value: np.ndarray):
    if value.dtype.char == "U":
        value = value.astype("S")

    f.create_dataset(
        key,
        shape=value.shape,
        dtype=value.dtype,
        data=value,
        compression="gzip",
        chunks=True,
    )


if __name__ == "__main__":
    setup, mesh = smash.factory.load_dataset("Cance")

    model = smash.Model(setup, mesh)

    model.forward_run()

    qs = model.sim_response.q[:]

    if os.path.exists("new_simulated_discharges.hdf5"):
        os.remove("new_simulated_discharges.hdf5")

    if not os.path.exists("simulated_discharges.hdf5"):
        file_name = "simulated_discharges.hdf5"
    else:
        file_name = "new_simulated_discharges.hdf5"

    with h5py.File(file_name, "w") as f:
        f.create_dataset(
            "sim_q",
            shape=qs.shape,
            dtype=qs.dtype,
            data=qs,
            compression="gzip",
            chunks=True,
        )
