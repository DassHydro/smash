from __future__ import annotations

import os

import numpy as np
import pytest

import smash


def generic_custom_bayesian_optimize(model: smash.Model, **kwargs) -> dict:
    res = {}

    ncpu = min(5, max(1, os.cpu_count() - 1))

    custom_sets = [
        # % Test custom optimize_options
        {
            "mapping": "distributed",
            "optimizer": "lbfgsb",
            "optimize_options": {
                "parameters": ["cp", "ct", "kexc", "llr", "sg0"],
                "bounds": {"cp": (10, 500), "llr": (1, 500), "sg0": (1e-6, 1e2)},
                "termination_crit": {"maxiter": 1},
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
                "termination_crit": {"maxiter": 1},
            },
            "common_options": {
                "ncpu": ncpu,
                "verbose": False,
            },
        },
        # Test custom cost_options
        {
            "cost_options": {
                "gauge": "dws",
            },
            "optimize_options": {
                "termination_crit": {"maxiter": 1},
            },
            "common_options": {
                "ncpu": ncpu,
                "verbose": False,
            },
        },
        {
            "cost_options": {
                "gauge": "all",
                "control_prior": {
                    "cp0": ["Gaussian", [200, 100]],
                    "kexc0": ["Gaussian", [0, 5]],
                },
            },
            "optimize_options": {
                "termination_crit": {"maxiter": 1},
            },
            "common_options": {
                "ncpu": ncpu,
                "verbose": False,
            },
        },
    ]

    for i, inner_kwargs in enumerate(custom_sets):
        instance = smash.bayesian_optimize(model, **inner_kwargs)

        qsim = instance.response.q[:].flatten()
        qsim = qsim[::10]  # extract values at every 10th position

        res[f"custom_bayesian_optimize.{model.setup.structure}.custom_set_{i+1}.sim_q"] = qsim

    return res


def test_custom_bayesian_optimize():
    res = generic_custom_bayesian_optimize(pytest.model)

    for key, value in res.items():
        # % Check qsim in sparse storage run
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-03), key
