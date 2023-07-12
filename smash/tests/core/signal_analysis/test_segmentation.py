from __future__ import annotations

import smash

import numpy as np
import pytest


def generic_hydrograph_segmentation(model: smash.Model, **kwargs) -> dict:
    arr = smash.hydrograph_segmentation(model).to_numpy()

    res = {"hydrograph_segmentation.arr": arr.astype("S")}

    return res


def test_hydrograph_segmentation():
    res = generic_hydrograph_segmentation(pytest.model)

    for key, value in res.items():
        # % Check hydrograph segmentation res
        assert np.array_equal(value, pytest.baseline[key][:]), key
