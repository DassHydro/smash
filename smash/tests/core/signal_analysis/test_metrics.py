from __future__ import annotations

import numpy as np
import pytest

import smash
from smash._constant import METRICS


def generic_metrics(model: smash.Model, qs: np.ndarray, **kwargs) -> dict:
    res = {}

    instance = model.copy()

    instance.response.q = qs

    for metric in METRICS:
        res[f"metrics.{metric}"] = smash.metrics(instance, metric=metric)

    return res


def test_metrics():
    res = generic_metrics(pytest.model, pytest.simulated_discharges["sim_q"][:])

    for key, value in res.items():
        # % Check hydrograph segmentation res
        assert np.array_equal(value, pytest.baseline[key][:]), key
