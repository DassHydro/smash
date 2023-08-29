from __future__ import annotations

import smash

import numpy as np
import pytest


def generic_precipitation_indices(model: smash.Model, **kwargs) -> dict:
    res = {
        "precipitation_indices." + k: v
        for k, v in smash.precipitation_indices(model).items()
    }

    return res


def test_precipitation_indices():
    res = generic_precipitation_indices(pytest.model)

    for key, value in res.items():
        # % Check precipitation indices
        assert np.allclose(
            value,
            pytest.baseline[key][:],
            equal_nan=True,
            atol=1e-06,
        ), key


def test_sparse_precipitation_indices():
    res = generic_precipitation_indices(pytest.sparse_model)

    for key, value in res.items():
        # % Check precipitation indices in sparse storage
        assert np.allclose(
            value,
            pytest.baseline[key][:],
            equal_nan=True,
            atol=1e-06,
        ), (
            "sparse." + key
        )
