from __future__ import annotations

import numpy as np
import pytest

import smash


def generic_mesh_io(**kwargs) -> dict:
    setup, mesh = smash.factory.load_dataset("Cance")

    smash.io.save_mesh(mesh, "tmp_mesh.hdf5")

    mesh_rld = smash.io.read_mesh("tmp_mesh.hdf5")

    res = {"mesh_io." + k: np.array(v, ndmin=1) for (k, v) in mesh_rld.items()}

    return res


def test_mesh_io():
    res = generic_mesh_io()

    for key, value in res.items():
        if value.dtype.char == "U":
            value = value.astype("S")
            # % Check all values read from mesh
            assert np.array_equal(value, pytest.baseline[key][:]), key
        else:
            # % Check all values read from mesh
            assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key
