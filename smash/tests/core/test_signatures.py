from smash.core._constant import STRUCTURE_PARAMETERS, CSIGN, ESIGN

import numpy as np

import pytest


def test_signatures():

    instance = pytest.model.copy()
    instance.run(inplace=True)

    signresult = instance.signatures()

    for typ, sign in zip(
        ["cont", "event"], [CSIGN[:4], ESIGN]
    ):  # remove percentile signatures calculation

        for dom in ["obs", "sim"]:

            arr = signresult[typ][dom][sign].to_numpy(dtype=np.float32)

            assert np.allclose(
                arr,
                pytest.baseline[f"signatures.{typ}_{dom}"][:],
                equal_nan=True,
                atol=1e-06,
            )


def test_signatures_sens():

    signsensresult = pytest.model.signatures_sensitivity(n=8, random_state=11)

    for typ, sign in zip(
        ["cont", "event"], [CSIGN[:4], ESIGN]
    ):  # remove percentile signatures calculation

        for ordr in ["first_si", "total_si"]:

            for param in STRUCTURE_PARAMETERS[pytest.model.setup.structure]:

                arr = signsensresult[typ][ordr][param][sign].to_numpy(dtype=np.float32)

                assert np.allclose(
                    arr,
                    pytest.baseline[f"signatures_sens.{typ}_{ordr}_{param}"][:],
                    equal_nan=True,
                    atol=1e-03,
                )
