from __future__ import annotations

import smash

import numpy as np
import pytest


def generic_model_io(**kwargs) -> dict:
    setup, mesh = smash.factory.load_dataset("Cance")

    model = smash.Model(setup, mesh)

    smash.io.save_model(model, "tmp_model.hdf5")

    model_rld = smash.io.read_model("tmp_model.hdf5")

    res = {}

    dict_check = {
        "setup": ["structure", "start_time", "end_time"],
        "mesh": ["flwdir", "active_cell"],
        "response_data": ["q"],
        "response": ["q"],
        "physio_data": ["descriptor"],
        "atmos_data": ["mean_prcp", "mean_pet"],
        "opr_parameters": ["values"],
        "opr_initial_states": ["values"],
        "opr_final_states": ["values"],
    }

    for attr, list_sub_attrs in dict_check.items():
        for sub_attr in list_sub_attrs:
            value = getattr(getattr(model_rld, attr), sub_attr)

            if isinstance(value, str):
                value = np.array(value, ndmin=1)

            elif isinstance(value, np.ndarray):
                value = value.flatten()[::50]

            res[f"model_io.{attr}.{sub_attr}"] = value

    return res


def test_model_io():
    res = generic_model_io()

    for key, value in res.items():
        if value.dtype.char == "U":
            value = value.astype("S")
            # % Check all values read from model
            assert np.array_equal(value, pytest.baseline[key][:]), key
        else:
            # % Check all values read from model
            assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key
