from __future__ import annotations

import os

import numpy as np
import pytest

import smash


def generic_multiset_estimate(model: smash.Model, **kwargs) -> dict:
    res = {}

    ncpu = min(5, max(1, os.cpu_count() - 1))

    problem = {
        "num_vars": 3,
        "names": ["cp", "kexc", "hp"],
        "bounds": [(20, 1000), (-20, 5), (0.2, 0.8)],
    }

    sample = smash.factory.generate_samples(problem, n=12, random_state=1)

    multiset_1 = smash.multiple_forward_run(model, sample, common_options={"ncpu": ncpu, "verbose": False})

    spl_sample = {k: [] for k in problem["names"]}
    for i in range(sample.n_sample):
        instance = model.copy()

        for k in problem["names"]:
            if k in instance.rr_parameters.keys:
                instance.set_rr_parameters(k, getattr(sample, k)[i])
            elif k in instance.rr_initial_states.keys:
                instance.set_rr_initial_states(k, getattr(sample, k)[i])

        instance.optimize(
            mapping="uniform",
            optimize_options={"parameters": problem["names"], "termination_crit": {"maxiter": 1}},
            common_options={"ncpu": ncpu, "verbose": False},
        )

        for k in problem["names"]:
            try:
                value = instance.get_rr_parameters(k).copy()
            except ValueError:
                value = instance.get_rr_initial_states(k).copy()
            spl_sample[k].append(value)

        del instance

    multiset_2 = smash.multiple_forward_run(
        model, spl_sample, common_options={"ncpu": ncpu, "verbose": False}
    )

    for i, multiset in enumerate([multiset_1, multiset_2]):
        instance, ret = smash.multiset_estimate(
            model,
            multiset,
            alpha=np.linspace(-1, 6, 10),
            common_options={"ncpu": ncpu, "verbose": False},
            return_options={"lcurve_multiset": True},
        )

        for key, value in ret.lcurve_multiset.items():
            res[f"multiset_estimate.set_{i+1}.lcurve_multiset.{key}"] = np.array(value, ndmin=1)

        qsim = instance.response.q[:].flatten()
        qsim = qsim[::10]  # extract values at every 10th position

        res[f"multiset_estimate.set_{i+1}.sim_q"] = qsim

    return res


def test_multiset_estimate():
    res = generic_multiset_estimate(pytest.model)

    for key, value in res.items():
        # % Check qsim in run
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-03, equal_nan=True), key
