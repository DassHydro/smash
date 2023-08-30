from __future__ import annotations

import smash

from smash._constant import MODEL_PUBLIC_ATTRS

import numpy as np
import pytest


def generic_model_io(**kwargs) -> dict:
    setup, mesh = smash.factory.load_dataset("Cance")

    model = smash.Model(setup, mesh)

    smash.io.save_model(model, "tmp_model.hdf5")

    model_rld = smash.io.read_model("tmp_model.hdf5")

    res = {
        "model_io." + attr: np.array(dir(getattr(model_rld, attr))).astype("S")
        for attr in MODEL_PUBLIC_ATTRS
    }

    return res


def test_model_io():
    res = generic_model_io()

    for key, value in res.items():
        # % Check all values read from model
        assert np.array_equal(value, pytest.baseline[key][:]), key
