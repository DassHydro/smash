from __future__ import annotations

import numpy as np
import pytest

import smash


def generic_setup_io(**kwargs) -> dict:
    setup, mesh = smash.factory.load_dataset("Cance")

    smash.io.save_setup(setup, "tmp_setup.yaml")

    setup_rld = smash.io.read_setup("tmp_setup.yaml")

    res = {"setup_io." + k: np.array(v, ndmin=1) for (k, v) in setup_rld.items() if "directory" not in k}

    return res


def test_setup_io():
    res = generic_setup_io()

    for key, value in res.items():
        if value.dtype.char == "U":
            value = value.astype("S")

        # % Check all values read from setup
        assert np.array_equal(value, pytest.baseline[key][:]), key
