from __future__ import annotations

import smash

import numpy as np
import pytest


def generic_hydrograph_segmentation(model: smash.Model, **kwargs) -> dict:
    instance = smash.forward_run(model)

    by_obs = smash.hydrograph_segmentation(instance, by="obs").to_numpy()
    by_sim = smash.hydrograph_segmentation(instance, by="sim").to_numpy()

    res = {
        "hydrograph_segmentation.by_obs": by_obs.astype("S"),
        "hydrograph_segmentation.by_sim": by_sim.astype("S"),
    }

    return res


def test_hydrograph_segmentation():
    res = generic_hydrograph_segmentation(pytest.model)

    for key, value in res.items():
        # % Check hydrograph segmentation res
        assert np.array_equal(value, pytest.baseline[key][:]), key
