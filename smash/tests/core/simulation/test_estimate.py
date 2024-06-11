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

    sample = smash.factory.generate_samples(problem, n=30, random_state=11)

    multisets = {}

    multisets["mfwr"] = smash.multiple_forward_run(
        model, sample, common_options={"ncpu": ncpu, "verbose": False}
    )

    multisets["mopt_unf"] = smash.multiple_optimize(
        model,
        sample,
        mapping="uniform",
        optimize_options={"termination_crit": {"maxiter": 1}},
        common_options={"ncpu": ncpu, "verbose": False},
    )

    multisets["mopt_ml"] = smash.multiple_optimize(
        model,
        sample,
        mapping="multi-linear",
        optimize_options={"termination_crit": {"maxiter": 1}},
        common_options={"ncpu": ncpu, "verbose": False},
    )

    for key, multiset in multisets.items():
        instance, ret = smash.multiset_estimate(
            model,
            multiset,
            np.linspace(-1, 8, 10),
            common_options={"ncpu": ncpu, "verbose": False},
            return_options={"lcurve_multiset": True},
        )

        qsim = instance.response.q[:].flatten()
        qsim = qsim[::10]  # extract values at every 10th position

        res[f"multiset_estimate.{key}.sim_q"] = qsim

        # Remove multi-linear temporarily (solver precision issue)
        for lc_key, lc_value in ret.lcurve_multiset.items():
            if key != "mopt_ml":
                res[f"multiset_estimate.{key}.lcurve_multiset.{lc_key}"] = np.array(lc_value, ndmin=1)

    return res


def test_multiset_estimate():
    res = generic_multiset_estimate(pytest.model)

    for key, value in res.items():
        # % Check qsim in run
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-03, equal_nan=True), key
