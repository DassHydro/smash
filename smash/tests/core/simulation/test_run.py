from __future__ import annotations

import os

import numpy as np
import pytest

import smash


def generic_forward_run(model_structure: list[smash.Model], **kwargs) -> dict:
    res = {}

    ncpu = min(5, max(1, os.cpu_count() - 1))

    for model in model_structure:
        # % There is no snow data for the Cance dataset.
        # % TODO: Add a dataset to test snow module
        if model.setup.snow_module_present:
            continue

        # % Hybrid forward hydrological model with NN
        if model.setup.n_layers > 0:
            model.set_nn_parameters_weight(initializer="glorot_normal", random_state=11)

        instance, ret = smash.forward_run(
            model,
            common_options={"verbose": False, "ncpu": ncpu},
            return_options={
                "cost": True,
                "jobs": True,
                "q_domain": True,
                "rr_states": True,
            },
        )

        mask = model.mesh.active_cell == 0

        qsim = instance.response.q[:].flatten()
        qsim = qsim[qsim > np.quantile(qsim, 0.95)]  # extract values depassing 0.95-quantile

        res[f"forward_run.{instance.setup.structure}.sim_q"] = qsim
        res[f"forward_run.{instance.setup.structure}.cost"] = np.array(ret.cost, ndmin=1)
        res[f"forward_run.{instance.setup.structure}.jobs"] = np.array(ret.jobs, ndmin=1)
        res[f"forward_run.{instance.setup.structure}.q_domain"] = np.where(
            mask, np.nan, ret.q_domain[..., -1]
        )
        for i, key in enumerate(model.rr_initial_states.keys):
            res[f"forward_run.{instance.setup.structure}.rr_states.{key}"] = np.where(
                mask, np.nan, ret.rr_states[-1].values[..., i]
            )

    return res


def test_forward_run():
    res = generic_forward_run(pytest.model_structure)

    for key, value in res.items():
        # % Check qsim in run
        if key.split(".")[-1] in ("sim_q", "q_domain"):
            atol = 1e-02  # sim_q and q_domain with high tolerance
        else:
            atol = 1e-06

        assert np.allclose(value, pytest.baseline[key][:], atol=atol, equal_nan=True), key


def test_sparse_forward_run():
    res = generic_forward_run(pytest.sparse_model_structure)

    for key, value in res.items():
        # % Check qsim in sparse storage run
        if key.split(".")[-1] in ("sim_q", "q_domain"):
            atol = 1e-02  # sim_q and q_domain with high tolerance
        else:
            atol = 1e-06

        assert np.allclose(value, pytest.baseline[key][:], atol=atol, equal_nan=True), "sparse." + key


def test_multiple_forward_run():
    instance = pytest.model.copy()
    ncpu = min(5, max(1, os.cpu_count() - 1))

    problem = {
        "num_vars": 5,
        "names": ["cp", "ct", "kexc", "llr", "hp"],
        "bounds": [(1, 1_000), (1, 1_000), (-20, 5), (1, 200), (0.1, 0.9)],
    }
    n_sample = 5
    samples = smash.factory.generate_samples(problem, random_state=99, n=n_sample)

    frq = np.zeros(shape=(*instance.response_data.q.shape, n_sample), dtype=np.float32)

    for i in range(n_sample):
        for key in samples._problem["names"]:
            if key in instance.rr_parameters.keys:
                instance.set_rr_parameters(key, getattr(samples, key)[i])
            elif key in instance.rr_initial_states.keys:
                instance.set_rr_initial_states(key, getattr(samples, key)[i])
        instance.forward_run(common_options={"ncpu": ncpu})
        frq[..., i] = instance.response.q.copy()

    mfr = smash.multiple_forward_run(instance, samples, common_options={"verbose": False, "ncpu": ncpu})

    # % Check that forward run discharge is equivalent to multiple forward run discharge
    assert np.allclose(frq, mfr.q, atol=1e-06, equal_nan=True), "multiple_forward_run.q"


def test_forward_run_mlp():
    ncpu = min(5, max(1, os.cpu_count() - 1))

    # % Test some multi layer perceptron models versus classical models
    mlp_to_cls_structure = {
        "zero-gr4_mlp-lr": "zero-gr4-lr",
        "zero-gr4_ode_mlp-lr": "zero-gr4_ode-lr",
        "zero-gr5_mlp-lr": "zero-gr5-lr",
        "zero-gr6_mlp-lr": "zero-gr6-lr",
        "zero-grd_mlp-lr": "zero-grd-lr",
        "zero-grc_mlp-lr": "zero-grc-lr",
        "zero-loieau_mlp-lr": "zero-loieau-lr",
    }
    all_structure = [model.setup.structure for model in pytest.model_structure]
    for mlp_structure, cls_structure in mlp_to_cls_structure.items():
        mlp_idx = all_structure.index(mlp_structure)
        cls_idx = all_structure.index(cls_structure)
        mlp_model = pytest.model_structure[mlp_idx]
        cls_model = pytest.model_structure[cls_idx]

        # % Reset NN param values to zeros
        mlp_model.set_nn_parameters_weight(initializer="zeros")
        mlp_model.set_nn_parameters_bias(initializer="zeros")

        mlp_instance, mlp_ret = smash.forward_run(
            mlp_model,
            common_options={"verbose": False, "ncpu": ncpu},
            return_options={
                "cost": True,
            },
        )
        cls_instance, cls_ret = smash.forward_run(
            cls_model,
            common_options={"verbose": False, "ncpu": ncpu},
            return_options={
                "cost": True,
            },
        )
        assert np.allclose(
            mlp_instance.response.q[:], cls_instance.response.q[:], atol=1e-06, equal_nan=True
        ), f"forward_run_mlp.{mlp_model.setup.structure}.q"
        assert np.allclose(mlp_ret.cost, cls_ret.cost, atol=1e-06, equal_nan=True), (
            f"forward_run_mlp.{mlp_model.setup.structure}.cost"
        )
