from __future__ import annotations

import numpy as np
import pytest

import smash
from smash._constant import METRICS


def generic_metrics(model: smash.Model, qs: np.ndarray, **kwargs) -> dict:
    res = {}

    instance = model.copy()

    instance.response.q = qs

    metrics = smash.metrics(instance, criteria=METRICS)

    for i, m in enumerate(METRICS):
        res[f"metrics.{m}"] = metrics[:, i]

    return res


def test_metrics():
    res = generic_metrics(pytest.model, pytest.simulated_discharges["sim_q"][:])

    for key, value in res.items():
        # % Check hydrograph segmentation res
        assert np.array_equal(value, pytest.baseline[key][:]), key
