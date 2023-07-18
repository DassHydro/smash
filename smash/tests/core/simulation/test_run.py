from __future__ import annotations

import smash

import numpy as np
import pytest


def generic_forward_run(model_structure: list[smash.Model], **kwargs) -> dict:
    res = {}

    for model in model_structure:
        instance = smash.forward_run(model)

        qsim = instance.sim_response.q[:].flatten()
        qsim = qsim[::10]  # extract values at every 10th position

        res[f"forward_run.{instance.setup.structure}.sim_q"] = qsim

    return res


def test_forward_run():
    res = generic_forward_run(pytest.model_structure)

    for key, value in res.items():
        # % Check qsim in run
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key


def test_sparse_forward_run():
    res = generic_forward_run(pytest.sparse_model_structure)

    for key, value in res.items():
        # % Check qsim in sparse storage run
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), "sparse." + key
