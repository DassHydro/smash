from __future__ import annotations

import smash

from smash._constant import METRICS

import numpy as np
import pytest


def generic_metrics(model: smash.Model, **kwargs) -> dict:
    res = {}

    instance = smash.forward_run(model)

    for metric in METRICS:
        res[f"metrics.{metric}"] = smash.metrics(instance, metric=metric)

    return res


def test_metrics():
    res = generic_metrics(pytest.model)

    for key, value in res.items():
        # % Check hydrograph segmentation res
        assert np.array_equal(value, pytest.baseline[key][:]), key
