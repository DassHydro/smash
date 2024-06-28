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
                instance, ret = smash.optimize(
                    model,
                    mapping=mp,
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


def test_multiple_optimize():
    instance = pytest.model.copy()
    ncpu = min(5, max(1, os.cpu_count() - 1))

    problem = {
        "num_vars": 5,
        "names": ["cp", "ct", "kexc", "llr", "hp"],
        "bounds": [(1, 1_000), (1, 1_000), (-20, 5), (1, 200), (0.1, 0.9)],
    }
    n_sample = 5
    samples = smash.factory.generate_samples(problem, random_state=99, n=n_sample)

    optq = np.zeros(shape=(*instance.response_data.q.shape, n_sample), dtype=np.float32)

    for i in range(n_sample):
        for key in samples._problem["names"]:
            if key in instance.rr_parameters.keys:
                instance.set_rr_parameters(key, getattr(samples, key)[i])
            elif key in instance.rr_initial_states.keys:
                instance.set_rr_initial_states(key, getattr(samples, key)[i])
        instance.optimize(
            mapping="distributed",
            optimize_options={"termination_crit": {"maxiter": 1}},
            common_options={"verbose": False, "ncpu": ncpu},
        )
        optq[..., i] = instance.response.q.copy()

    mopt = smash.multiple_optimize(
        instance,
        samples,
        mapping="distributed",
        optimize_options={"termination_crit": {"maxiter": 1}},
        common_options={"verbose": False, "ncpu": ncpu},
    )

    # % Check that optimize discharge is equivalent to multiple optimize discharge
    assert np.allclose(optq, mopt.q, atol=1e-03, equal_nan=True), "multiple_optimize.q"
