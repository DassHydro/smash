from __future__ import annotations

import smash

import numpy as np
import pytest


def generic_xy_mesh(**kwargs) -> dict:
    flwdir = smash.load_dataset("flwdir")

    mesh = smash.generate_mesh(
        flwdir,
        x=[840_261, 826_553, 828_269],
        y=[6_457_807, 6_467_115, 6_469_198],
        area=[381.7 * 1e6, 107 * 1e6, 25.3 * 1e6],
        code=["V3524010", "V3515010", "V3517010"],
        epsg=2154,
    )

    res = {
        "xy_mesh.flwdir": mesh["flwdir"],
        "xy_mesh.flwdst": mesh["flwdst"],
        "xy_mesh.flwacc": mesh["flwacc"],
    }

    return res


def test_xy_mesh():
    res = generic_xy_mesh()

    for key, value in res.items():
        # % Check xy_mesh flwdir, flwdst and flwacc
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key


def generic_bbox_mesh(**kwargs) -> dict:
    flwdir = smash.load_dataset("flwdir")

    mesh = smash.generate_mesh(
        flwdir,
        bbox=(100_000, 1_250_000, 6_050_000, 7_125_000),
        epsg=2154,
    )

    res = {
        "bbox_mesh.flwdir": mesh["flwdir"],
        "bbox_mesh.flwacc": mesh["flwacc"],
    }

    return res


def test_bbox_mesh():
    res = generic_bbox_mesh()

    for key, value in res.items():
        # % Check bbox_mesh flwdir, flwdst and flwacc
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key
