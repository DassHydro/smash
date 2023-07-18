from __future__ import annotations

import smash

import numpy as np
import pytest


def generic_xy_mesh(**kwargs) -> dict:
    flwdir = smash.factory.load_dataset("flwdir")

    mesh = smash.factory.generate_mesh(
        flwdir,
        x=[840_261, 826_553, 828_269],
        y=[6_457_807, 6_467_115, 6_469_198],
        area=[381.7 * 1e6, 107 * 1e6, 25.3 * 1e6],
        code=["V3524010", "V3515010", "V3517010"],
        epsg=2154,
    )
    # % Remove 'path' check because of updated numpy argsort method
    skip_keys = ["path"]
    res = {
        "xy_mesh." + k: np.array(v, ndmin=1)
        for (k, v) in mesh.items()
        if k not in skip_keys
    }

    return res


def test_xy_mesh():
    res = generic_xy_mesh()

    for key, value in res.items():
        if value.dtype.char == "U":
            value = value.astype("S")
            # % Check xy_mesh
            assert np.array_equal(value, pytest.baseline[key][:]), key
        else:
            # % Check xy_mesh
            assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key


def generic_bbox_mesh(**kwargs) -> dict:
    flwdir = smash.factory.load_dataset("flwdir")

    mesh = smash.factory.generate_mesh(
        flwdir,
        bbox=(100_000, 200_000, 6_050_000, 6_150_000),
        epsg=2154,
    )
    # % Remove 'path' check because of updated numpy argsort method
    skip_keys = ["path"]
    res = {
        "bbox_mesh." + k: np.array(v, ndmin=1)
        for (k, v) in mesh.items()
        if k not in skip_keys
    }

    return res


def test_bbox_mesh():
    res = generic_bbox_mesh()

    for key, value in res.items():
        if value.dtype.char == "U":
            value = value.astype("S")
            # % Check bbox_mesh
            assert np.array_equal(value, pytest.baseline[key][:]), key
        else:
            # % Check bbox_mesh
            assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key
