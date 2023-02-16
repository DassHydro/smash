from __future__ import annotations

import smash
from smash.core._constant import STRUCTURE_PARAMETERS, CSIGN, ESIGN

import numpy as np
import pytest


def generic_signatures(model: smash.Model, **kwargs) -> dict:
    res = {}

    instance = model.run()

    signresult = instance.signatures()

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


def generic_signatures_sens(model: smash.Model, **kwargs) -> dict:
    res = {}

    signsensresult = model.signatures_sensitivity(n=8, random_state=11)

    for typ, sign in zip(
        ["cont", "event"], [CSIGN[:4], ESIGN]
    ):  # % remove percentile signatures calculation
        for ordr in ["first_si", "total_si"]:
            for param in STRUCTURE_PARAMETERS[model.setup.structure]:
                res[f"signatures_sens.{typ}_{ordr}_{param}"] = signsensresult[typ][
                    ordr
                ][param][sign].to_numpy(dtype=np.float32)

    return res


def test_signatures_sens():
    res = generic_signatures_sens(pytest.model)

    for key, value in res.items():
        # % Check signatures for cont/event and obs/sim
        # % less accurate with a small value of n=8 -> atol=1e-3
        assert np.allclose(
            value,
            pytest.baseline[key][:],
            equal_nan=True,
            atol=1e-03,
        ), key
