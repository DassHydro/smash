from __future__ import annotations

import smash
from smash._constant import MAPPING

import numpy as np
import pytest
import os


def generic_optimize(model_structure: list[smash.Model], **kwargs) -> dict:
    res = {}

    ncpu = max(1, os.cpu_count() - 1)

    for model in model_structure:
        for mp in MAPPING:
            # % TODO: Temporary condition until ANN optimization is implemented
            if mp != "ann":
                instance = smash.optimize(
                    model,
                    mapping=mp,
                    optimize_options={"maxiter": 1},
                    common_options={"ncpu": ncpu, "verbose": False},
                )

                qsim = instance.sim_response.q[:].flatten()
                qsim = qsim[::10]  # extract values at every 10th position

                res[f"optimize.{model.setup.structure}.{mp}.sim_q"] = qsim

    return res


def test_optimize():
    res = generic_optimize(pytest.model_structure)

    for key, value in res.items():
        # % Check qsim in run
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-05), key


def test_sparse_optimize():
    res = generic_optimize(pytest.sparse_model_structure)

    for key, value in res.items():
        # % Check qsim in sparse storage run
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-05), "sparse." + key


def generic_custom_optimize(model: smash.Model, **kwargs) -> dict:
    res = {}

    ncpu = max(1, os.cpu_count() - 1)

    # % TODO: Add custom cost_options when cost computation is implemented
    custom_sets = [
        {
            "mapping": "distributed",
            "optimizer": "lbfgsb",
            "optimize_options": {
                "parameters": ["cp", "ct", "kexc", "llr", "hp"],
                "bounds": {"cp": (10, 500), "llr": (1, 500), "hp": (1e-6, 0.8)},
                "maxiter": 1,
            },
            "common_options": {
                "ncpu": ncpu,
                "verbose": False,
            },
        },
        {
            "mapping": "uniform",
            "optimizer": "sbs",
            "optimize_options": {
                "parameters": ["cp", "ct", "llr"],
                "bounds": {"cp": (10, 500), "llr": (1, 500)},
                "maxiter": 1,
            },
            "common_options": {
                "ncpu": ncpu,
                "verbose": False,
            },
        },
        {
            "mapping": "multi-linear",
            "optimizer": "lbfgsb",
            "optimize_options": {
                "parameters": ["cp", "ct"],
                "bounds": {"cp": (10, 2000)},
                "control_tfm": None,
                "descriptor": {"cp": "slope", "ct": "dd"},
                "maxiter": 1,
            },
            "common_options": {
                "ncpu": ncpu,
                "verbose": False,
            },
        },
        {
            "mapping": "multi-polynomial",
            "optimizer": "lbfgsb",
            "optimize_options": {
                "parameters": ["cp", "kexc"],
                "bounds": {"cp": (10, 2000), "kexc": (-50, 20)},
                "maxiter": 1,
            },
            "common_options": {
                "ncpu": ncpu,
                "verbose": False,
            },
        },
    ]

    for i, kwargs in enumerate(custom_sets):
        instance = smash.optimize(model, **kwargs)

        qsim = instance.sim_response.q[:].flatten()
        qsim = qsim[::10]  # extract values at every 10th position

        res[f"custom_optimize.{model.setup.structure}.custom_set_{i+1}.sim_q"] = qsim

    return res


def test_custom_optimize():
    res = generic_custom_optimize(pytest.model)

    for key, value in res.items():
        # % Check qsim in sparse storage run
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-05), key
