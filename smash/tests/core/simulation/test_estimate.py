from __future__ import annotations

import smash
from smash._constant import MAPPING

import numpy as np
import pytest
import os


def generic_multiset_estimate(model: smash.Model, **kwargs) -> dict:
    res = {}

    ncpu = max(1, os.cpu_count() - 1)

    problem = {
        "num_vars": 3,
        "names": ["cp", "kexc", "hp"],
        "bounds": [(1, 500), (-50, 50), (0, 1)],
    }

    sample = smash.factory.generate_samples(problem, n=20, random_state=11)

    multiset = smash.multiple_optimize(
        model,
        sample,
        mapping="multi-linear",
        optimize_options={"termination_crit": {"maxiter": 1}},
        common_options={"ncpu": ncpu, "verbose": False},
    )

    instance = smash.multiset_estimate(
        model,
        multiset,
        np.linspace(-1, 8, 10),
        common_options={"ncpu": ncpu, "verbose": False},
    )

    qsim = instance.sim_response.q[:].flatten()
    qsim = qsim[::10]  # extract values at every 10th position

    res[f"multiset_estimate.sim_q"] = qsim

    return res


def test_multiset_estimate():
    res = generic_multiset_estimate(pytest.model)

    for key, value in res.items():
        # % Check qsim in run
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-03), key
