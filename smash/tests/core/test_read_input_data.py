from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import smash


# % bbox (xmin, xmax, ymin, ymax)
def generate_minimal_mesh(bbox: tuple[float], xres: float, yres: float) -> dict[str, float]:
    mesh = {
        "xres": xres,
        "yres": yres,
        "xmin": bbox[0],
        "ymax": bbox[3],
        "nrow": (bbox[3] - bbox[2]) / yres,
        "ncol": (bbox[1] - bbox[0]) / xres,
        "npar": 0,
        "ng": 0,
    }

    return mesh


def test_missing_read_input_data():
    setup, _ = smash.factory.load_dataset("france")

    nts_off = 10
    prv_et = pd.Timestamp(setup["end_time"])
    et = prv_et + pd.DateOffset(hours=nts_off)
    setup["end_time"] = et

    bbox = (670_000, 770_000, 6_600_000, 6_700_000)
    mesh = generate_minimal_mesh(bbox, 1000, 1000)

    # % Missing files
    with pytest.warns(UserWarning, match=rf"Missing warning: missing {nts_off} file\(s\)"):
        model = smash.Model(setup, mesh)

    # % Check no data for missing files
    for i, date in enumerate(pd.date_range(setup["start_time"], setup["end_time"], freq="h")[1:]):
        if date > prv_et:
            assert np.all(
                model.atmos_data.prcp[..., i] < 0
            ), f"missing_read_input_data.prcp_partially_no_data_index_{i}"

    # Case where we will have only missing files
    setup["start_time"] = setup["end_time"]
    setup["end_time"] = setup["start_time"] + pd.DateOffset(hours=nts_off)

    # % Missing files
    with pytest.warns(UserWarning, match="Missing warning"):
        model = smash.Model(setup, mesh)

    # % Check no data for missing files
    assert np.all(model.atmos_data.prcp < 0), "missing_read_input_data.prcp_totally_no_data"


def test_resolution_read_input_data():
    setup, _ = smash.factory.load_dataset("france")

    bbox = (670_000, 770_000, 6_600_000, 6_700_000)

    mesh_hres = generate_minimal_mesh(bbox, 500, 500)
    mesh = generate_minimal_mesh(bbox, 1000, 1000)
    mesh_lres = generate_minimal_mesh(bbox, 2000, 2000)

    # % Higher resolution
    with pytest.warns(UserWarning, match="Resolution warning"):
        model_hres = smash.Model(setup, mesh_hres)

    # % Same resolution
    model = smash.Model(setup, mesh)

    # % Lower resolution
    with pytest.warns(UserWarning, match="Resolution warning"):
        smash.Model(setup, mesh_lres)

    # % Check mean values eq on higher resolution
    assert np.isclose(
        np.mean(model_hres.atmos_data.prcp), np.mean(model.atmos_data.prcp)
    ), "resolution_read_input_data.mean_prcp_hres_eq"

    assert np.isclose(
        np.mean(model_hres.atmos_data.pet), np.mean(model.atmos_data.pet)
    ), "resolution_read_input_data.mean_pet_hres_eq"


def test_overlap_read_input_data():
    setup, _ = smash.factory.load_dataset("france")

    xres, yres = 1000, 1000
    bbox = (670_000, 770_000, 6_600_000, 6_700_000)
    bbox_loff = (670_100, 770_100, 6_600_100, 6_700_100)
    bbox_roff = (670_900, 770_900, 6_600_900, 6_700_900)

    mesh = generate_minimal_mesh(bbox, xres, yres)
    mesh_loff = generate_minimal_mesh(bbox_loff, xres, yres)
    mesh_roff = generate_minimal_mesh(bbox_roff, xres, yres)

    # % Left offset
    with pytest.warns(UserWarning, match="Overlap warning"):
        model_loff = smash.Model(setup, mesh_loff)

    # % No offset
    model = smash.Model(setup, mesh)

    # % Right offset
    with pytest.warns(UserWarning, match="Overlap warning"):
        model_roff = smash.Model(setup, mesh_roff)

    # % Check mean values eq with left offset
    assert np.isclose(
        np.mean(model_loff.atmos_data.prcp), np.mean(model.atmos_data.prcp)
    ), "overlap_read_input_data.mean_prcp_loff_eq"

    assert np.isclose(
        np.mean(model_loff.atmos_data.pet), np.mean(model.atmos_data.pet)
    ), "overlap_read_input_data.mean_pet_loff_eq"

    # % Check mean values neq with right offset
    assert not np.isclose(
        np.mean(model_roff.atmos_data.prcp), np.mean(model.atmos_data.prcp)
    ), "overlap_read_input_data.mean_prcp_roff_neq"

    assert not np.isclose(
        np.mean(model_roff.atmos_data.pet), np.mean(model.atmos_data.pet)
    ), "overlap_read_input_data.mean_pet_roff_neq"


def test_outofbound_read_input_data():
    setup, _ = smash.factory.load_dataset("france")

    xres, yres = 1000, 1000
    bbox_poob = (670_000, 770_000, 7_100_000, 7_200_000)
    bbox_toob = (770_000, 770_000, 7_200_000, 7_300_000)

    mesh_poob = generate_minimal_mesh(bbox_poob, xres, yres)
    mesh_toob = generate_minimal_mesh(bbox_toob, xres, yres)

    # % Partially ouf of bound
    with pytest.warns(UserWarning, match="Out of bound warning: mesh is partially"):
        model_poob = smash.Model(setup, mesh_poob)

    # % Totally out of bound
    with pytest.warns(UserWarning, match="Out of bound warning: mesh is totally"):
        model_toob = smash.Model(setup, mesh_toob)

    # % Check no data values
    assert not np.all(model_poob.atmos_data.prcp < 0), "outofbound_read_input_data.prcp_partially_no_data"
    assert not np.all(model_poob.atmos_data.pet < 0), "outofbound_read_input_data.pet_partially_no_data"

    assert np.all(model_toob.atmos_data.prcp < 0), "outofbound_read_input_data.prcp_totally_no_data"
    assert np.all(model_toob.atmos_data.pet < 0), "outofbound_read_input_data.pet_totally_no_data"
