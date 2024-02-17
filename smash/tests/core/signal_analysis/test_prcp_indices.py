from __future__ import annotations

import numpy as np
import pytest

import smash
from smash._constant import PRECIPITATION_INDICES


def generic_precipitation_indices(model: smash.Model, **kwargs) -> dict:
    prcp_ind = smash.precipitation_indices(model)

    res = {"precipitation_indices." + k: getattr(prcp_ind, k) for k in PRECIPITATION_INDICES}

    return res


def test_precipitation_indices():
    res = generic_precipitation_indices(pytest.model)

    for key, value in res.items():
        # % Check precipitation indices
        assert np.allclose(
            value,
            pytest.baseline[key][:],
            equal_nan=True,
            atol=1e-04,
        ), key


def test_sparse_precipitation_indices():
    res = generic_precipitation_indices(pytest.sparse_model)

    for key, value in res.items():
        # % Check precipitation indices in sparse storage
        assert np.allclose(
            value,
            pytest.baseline[key][:],
            equal_nan=True,
            atol=1e-04,
        ), "sparse." + key
