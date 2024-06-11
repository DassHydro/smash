from __future__ import annotations

import numpy as np
import pytest

import smash


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
    res = {"xy_mesh." + k: np.array(v, ndmin=1) for (k, v) in mesh.items()}

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
    res = {"bbox_mesh." + k: np.array(v, ndmin=1) for (k, v) in mesh.items()}

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


def test_bbox_padding():
    flwdir = smash.factory.load_dataset("flwdir")

    bbox = (670_000, 770_000, 6_600_000, 6_700_000)
    bbox_loff = (670_100, 770_100, 6_600_100, 6_700_100)
    bbox_roff = (670_900, 770_900, 6_600_900, 6_700_900)

    mesh = smash.factory.generate_mesh(flwdir, bbox=bbox, epsg=2154)
    mesh_loff = smash.factory.generate_mesh(flwdir, bbox=bbox_loff, epsg=2154)
    mesh_roff = smash.factory.generate_mesh(flwdir, bbox=bbox_roff, epsg=2154)

    # % Check bounding box padding
    assert bbox[0] == mesh["xmin"], "overlap_read_data.nopad_xmin"
    assert bbox[3] == mesh["ymax"], "overlap_read_data.nopad_ymax"

    assert bbox_loff[0] != mesh_loff["xmin"], "overlap_read_data.loffpad_xmin"
    assert bbox_loff[3] != mesh_loff["ymax"], "overlap_read_data.loffpad_ymax"

    assert bbox_roff[0] != mesh_roff["xmin"], "overlap_read_data.uoffpad_xmin"
    assert bbox_roff[3] != mesh_roff["ymax"], "overlap_read_data.uoffpad_ymax"

    assert mesh_loff["xmin"] == mesh["xmin"], "overlap_read_data.loff_xmin_eq"
    assert mesh_loff["ymax"] == mesh["ymax"], "overlap_read_data.loff_ymax_eq"

    assert mesh_roff["xmin"] != mesh["xmin"], "overlap_read_data.uoff_xmin_neq"
    assert mesh_roff["ymax"] != mesh["ymax"], "overlap_read_data.uoff_ymax_neq"
