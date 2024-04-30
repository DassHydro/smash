from __future__ import annotations

import sys

import numpy as np
import pytest

import smash
from f90wrap.runtime import FortranDerivedTypeArray


def generic_model_io(**kwargs) -> dict:
    setup, mesh = smash.factory.load_dataset("Cance")

    # % Disable stdout
    # % TODO: replace this by adding a verbose argument at Model initialisation
    sys.stdout = open("/dev/null", "w")

    model = smash.Model(setup, mesh)

    # % Enable stdout
    sys.stdout = sys.__stdout__

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
        "rr_parameters": ["values"],
        "rr_initial_states": ["values"],
        "rr_final_states": ["values"],
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


def test_sparse_model_io_cmp():
    setup, mesh = smash.factory.load_dataset("Cance")
    setup["sparse_storage"] = True

    # % Disable stdout
    # % TODO: replace this by adding a verbose argument at Model initialisation
    sys.stdout = open("/dev/null", "w")

    model = smash.Model(setup, mesh)

    # % Enable stdout
    sys.stdout = sys.__stdout__

    smash.io.save_model(model, "tmp_model.hdf5")

    model_rld = smash.io.read_model("tmp_model.hdf5")

    dict_check = {
        "setup": ["structure", "start_time", "end_time"],
        "mesh": ["flwdir", "active_cell"],
        "response_data": ["q"],
        "response": ["q"],
        "physio_data": ["descriptor"],
        "atmos_data": ["mean_prcp", "mean_pet", "sparse_prcp", "sparse_pet"],
        "rr_parameters": ["values"],
        "rr_initial_states": ["values"],
        "rr_final_states": ["values"],
    }

    for attr, list_sub_attrs in dict_check.items():
        for sub_attr in list_sub_attrs:
            value = getattr(getattr(model, attr), sub_attr)
            value_rld = getattr(getattr(model_rld, attr), sub_attr)

            if isinstance(value, np.ndarray):
                assert np.allclose(value, value_rld, atol=1e-6), f"sparse_model_io_cmp.{sub_attr}"

            elif isinstance(value, FortranDerivedTypeArray):
                for i in range(len(value)):
                    for fdt_attr in dir(value[i]):
                        if fdt_attr.startswith("_"):
                            continue
                        try:
                            fdt_value = getattr(value[i], fdt_attr)
                            fdt_value_rld = getattr(value_rld[i], fdt_attr)
                            compare = True
                        except Exception:
                            compare = False

                        if callable(fdt_value):
                            continue

                        if compare:
                            if isinstance(fdt_value, np.ndarray):
                                assert np.allclose(
                                    fdt_value, fdt_value_rld, atol=1e-6
                                ), f"sparse_model_io_cmp.{sub_attr}.{fdt_attr}"
                            else:
                                assert (
                                    fdt_value == fdt_value_rld
                                ), f"sparse_model_io_cmp.{sub_attr}.{fdt_attr}"
            else:
                assert value == value_rld, f"sparse_model_io_cmp.{sub_attr}"
