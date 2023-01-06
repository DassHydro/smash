from smash.core._constant import STRUCTURE_PARAMETERS, CSIGN, ESIGN

import numpy as np

import pytest


def test_signatures():

    instance = pytest.model.copy()
    instance.run(inplace=True)

    signresult = instance.signatures()

    for typ, sign in zip(["cont", "event"], [CSIGN, ESIGN]):

        for dom in ["obs", "sim"]:

            arr = signresult[typ][dom][sign].to_numpy(dtype=np.float32)

            assert np.array_equal(arr, pytest.baseline[f"signatures.{typ}_{dom}"])


def test_signatures_sens():

    instance = pytest.model.copy()
    instance.run(inplace=True)

    signsensresult = instance.signatures_sensitivity(n=8, random_state=11)

    for typ, sign in zip(["cont", "event"], [CSIGN, ESIGN]):

        for ord in ["first_si", "total_si"]:

            for param in STRUCTURE_PARAMETERS[instance.setup.structure]:

                arr = signsensresult[typ][ord][param][sign].to_numpy(dtype=np.float32)

                assert np.array_equal(
                    arr, pytest.baseline[f"signatures_sens.{typ}_{ord}_{param}"]
                )
