from __future__ import annotations

import sys

import numpy as np
import pytest

import smash


def generic_model_ddt_io(**kwargs) -> dict:
    setup, mesh = smash.factory.load_dataset("Cance")

    # % Disable stdout
    # % TODO: replace this by adding a verbose argument at Model initialisation
    sys.stdout = open("/dev/null", "w")

    model = smash.Model(setup, mesh)

    # % Enable stdout
    sys.stdout = sys.__stdout__

    smash.io.save_model_ddt(model, "tmp_model_ddt.hdf5")

    model_dtt = smash.io.read_model_ddt("tmp_model_ddt.hdf5")

    res = {}

    for attr, value in model_dtt.items():
        for sub_attr, sub_value in value.items():
            if isinstance(sub_value, (str, int, float, np.number)):
                sub_value = np.array(sub_value, ndmin=1)
            elif isinstance(sub_value, np.ndarray):
                sub_value = sub_value.flatten()[::50]

            res[f"model_ddt_io.{attr}.{sub_attr}"] = sub_value

    return res


def test_model_ddt_io():
    res = generic_model_ddt_io()

    for key, value in res.items():
        if value.dtype.char == "U":
            value = value.astype("S")
            # % Check all values read from model
            assert np.array_equal(value, pytest.baseline[key][:]), key
        else:
            # % Check all values read from model
            assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key
