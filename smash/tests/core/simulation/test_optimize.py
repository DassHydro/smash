from __future__ import annotations

import os

import numpy as np
import pytest

import smash
from smash._constant import MAPPING


def generic_optimize(model_structure: list[smash.Model], **kwargs) -> dict:
    res = {}

    ncpu = min(5, max(1, os.cpu_count() - 1))

    for model in model_structure:
        # % There is no snow data for the Cance dataset.
        # % TODO: Add a dataset to test snow module
        if model.setup.snow_module_present:
            continue

        # % With VIC, remove ["cusl", "cbsl", "ks", "ds", "dsm"]
        if model.setup.hydrological_module == "vic3l":
            parameters = [
                key for key in model.rr_parameters.keys if key not in ["cusl", "cbsl", "ks", "ds", "dsm"]
            ]
        # % Else default parameters
        else:
            parameters = None

        # % Hybrid forward hydrological model with NN
        if model.setup.n_layers > 0:
            model.set_nn_parameters_weight(initializer="glorot_normal", random_state=11)

        for mp in MAPPING:
            if mp == "ann":
                instance = smash.optimize(
                    model,
                    mapping=mp,
                    optimize_options={
                        "parameters": parameters,
                        "learning_rate": 0.01,
                        "random_state": 11,
                        "termination_crit": {"epochs": 3},
                    },
                    common_options={"ncpu": ncpu, "verbose": False},
                )

            else:
                # Ignore SBS optimizer if the forward model uses NN
                opt = "lbfgsb" if model.setup.n_layers > 0 else None

                instance, ret = smash.optimize(
                    model,
                    mapping=mp,
                    optimizer=opt,
                    optimize_options={
                        "parameters": parameters,
                        "termination_crit": {"maxiter": 1},
                    },
                    common_options={"ncpu": ncpu, "verbose": False},
                    return_options={"iter_cost": True, "control_vector": True},
                )

                res[f"optimize.{model.setup.structure}.{mp}.iter_cost"] = ret.iter_cost
                res[f"optimize.{model.setup.structure}.{mp}.control_vector"] = ret.control_vector

            qsim = instance.response.q[:].flatten()
            qsim = qsim[::10]  # extract values at every 10th position

            res[f"optimize.{model.setup.structure}.{mp}.sim_q"] = qsim

    return res


def test_optimize():
    res = generic_optimize(pytest.model_structure)

    for key, value in res.items():
        # % Check qsim in run
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-03, equal_nan=True), key


def test_sparse_optimize():
    res = generic_optimize(pytest.sparse_model_structure)

    for key, value in res.items():
        # % Check qsim in sparse storage run
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-03, equal_nan=True), "sparse." + key


def generic_custom_optimize(model: smash.Model, **kwargs) -> dict:
    res = {}

    ncpu = min(5, max(1, os.cpu_count() - 1))

    custom_sets = [
        # % Test custom optimize_options
        {
            "mapping": "distributed",
            "optimizer": "lbfgsb",
            "optimize_options": {
                "parameters": ["cp", "ct", "kexc", "llr", "hp"],
                "bounds": {"cp": (10, 500), "llr": (1, 500), "hp": (1e-6, 0.8)},
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
        {
            "mapping": "multi-linear",
            "optimizer": "lbfgsb",
            "optimize_options": {
                "parameters": ["cp", "ct"],
                "bounds": {"cp": (10, 2000)},
                "control_tfm": None,
                "descriptor": {"cp": "slope", "ct": "dd"},
                "termination_crit": {"maxiter": 1},
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
                "jobs_cmpt": ["nse", "Crc", "Cfp10"],
                "wjobs_cmpt": "mean",
                "gauge": "all",
                "wgauge": [0.5, 0.3, 0.2],
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
                "jobs_cmpt": ["nse", "Epf", "Elt"],
                "wjobs_cmpt": [0.5, 1.5, 0.5],
                "event_seg": {"peak_quant": 0.9},
                "gauge": ["V3524010", "V3517010"],
                "wgauge": "uquartile",
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
                "jobs_cmpt": ["nse", "kge", "kge"],
                "jobs_cmpt_tfm": ["keep", "sqrt", "inv"],
                "wjobs_cmpt": "mean",
                "gauge": "all",
                "wgauge": [0.5, 0.3, 0.2],
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
                "wjreg": 0.0001,
                "jreg_cmpt": ["prior", "smoothing", "hard-smoothing"],
                "wjreg_cmpt": [0.5, 0.2, 0.2],
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
                "wjreg": "lcurve",
                "jreg_cmpt": ["prior", "smoothing"],
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
        instance = smash.optimize(model, **inner_kwargs)

        qsim = instance.response.q[:].flatten()
        qsim = qsim[::10]  # extract values at every 10th position

        res[f"custom_optimize.{model.setup.structure}.custom_set_{i+1}.sim_q"] = qsim

    return res


def test_custom_optimize():
    res = generic_custom_optimize(pytest.model)

    for key, value in res.items():
        # % Check qsim in sparse storage run
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-03), key
