from __future__ import annotations

import smash
from smash._constant import CSIGN, ESIGN

import numpy as np
import pytest


def generic_signatures(model: smash.Model, **kwargs) -> dict:
    res = {}

    instance = smash.forward_run(model)

    signresult = smash.signatures(instance)

    for typ, sign in zip(
        ["cont", "event"], [CSIGN[:4], ESIGN]
    ):  # % remove percentile signatures calculation
        for dom in ["obs", "sim"]:
            res[f"signatures.{typ}_{dom}"] = signresult[typ][dom][sign].to_numpy(
                dtype=np.float32
            )

    return res


def test_signatures():
    res = generic_signatures(pytest.model)

    for key, value in res.items():
        # % Check signatures for cont/event and obs/sim
        assert np.allclose(
            value,
            pytest.baseline[key][:],
            equal_nan=True,
            atol=1e-06,
        ), key
