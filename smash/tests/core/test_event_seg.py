from __future__ import annotations

import smash

import numpy as np
import pytest


def generic_event_seg(model: smash.Model, **kwargs) -> dict:
    arr = model.event_segmentation().to_numpy()

    res = {"event_seg.arr": arr.astype("S")}

    return res


def test_event_seg():
    res = generic_event_seg(pytest.model)

    for key, value in res.items():
        # % Check event segmentation res
        assert np.array_equal(value, pytest.baseline[key][:]), key
