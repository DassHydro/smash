from __future__ import annotations

import smash

import numpy as np
import pytest
import os


def generic_forward_run(model_structure: list[smash.Model], **kwargs) -> dict:
    res = {}

    ncpu = max(1, os.cpu_count() - 1)

    for model in model_structure:
        instance = smash.forward_run(model, common_options={"ncpu": ncpu})

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


def test_multiple_forward_run():
    instance = pytest.model.copy()
    ncpu = max(1, os.cpu_count() - 1)

    problem = {
        "num_vars": 5,
        "names": ["cp", "ct", "kexc", "llr", "hp"],
        "bounds": [(1, 1_000), (1, 1_000), (-20, 5), (1, 200), (0.1, 0.9)],
    }
    n_sample = 5
    samples = smash.factory.generate_samples(problem, random_state=99, n=n_sample)

    frq = np.zeros(shape=(*instance.obs_response.q.shape, n_sample), dtype=np.float32)

    for i in range(n_sample):
        for key in samples._problem["names"]:
            if key in instance.opr_parameters.keys:
                instance.set_opr_parameters(key, getattr(samples, key)[i])
            elif key in instance.opr_initial_states.keys:
                instance.set_opr_initial_states(key, getattr(samples, key)[i])
        instance.forward_run(common_options={"ncpu": ncpu})
        frq[..., i] = instance.sim_response.q.copy()

    mfr = smash.multiple_forward_run(
        instance, samples, common_options={"verbose": False, "ncpu": ncpu}
    )

    # % Check that forward run discharge is equivalent to multiple forward run discharge
    assert np.allclose(frq, mfr.q, atol=1e-06), "multiple_forward_run.q"
