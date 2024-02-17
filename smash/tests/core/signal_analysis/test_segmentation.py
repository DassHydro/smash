from __future__ import annotations

import numpy as np
import pytest

import smash


def generic_hydrograph_segmentation(model: smash.Model, qs: np.ndarray, **kwargs) -> dict:
    instance = model.copy()

    instance.response.q = qs

    by_obs = smash.hydrograph_segmentation(instance, by="obs").to_numpy()
    by_sim = smash.hydrograph_segmentation(instance, by="sim").to_numpy()

    res = {
        "hydrograph_segmentation.by_obs": by_obs.astype("S"),
        "hydrograph_segmentation.by_sim": by_sim.astype("S"),
    }

    return res


def test_hydrograph_segmentation():
    res = generic_hydrograph_segmentation(pytest.model, pytest.simulated_discharges["sim_q"][:])

    for key, value in res.items():
        # % Check hydrograph segmentation res
        assert np.array_equal(value, pytest.baseline[key][:]), key
