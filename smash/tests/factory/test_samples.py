from __future__ import annotations

import smash

from smash._constant import PROBLEM_KEYS

import numpy as np
import pytest


def generic_gen_samples(model: smash.Model, **kwargs) -> dict:
    bounds = model.get_opr_parameters_bounds()
    
    problem = dict(zip(PROBLEM_KEYS, [len(bounds), list(bounds.keys()), list(bounds.values())]))

    sample = smash.factory.generate_samples(problem, generator="uniform", n=20, random_state=11)
    uni = sample.to_numpy(axis=-1)

    sample = smash.factory.generate_samples(
        problem,
        generator="normal",
        n=20,
        mean={problem["names"][1]: 1 / 3 * np.mean(problem["bounds"][1])},
        coef_std=2,
        random_state=11,
    )
    nor = sample.to_numpy(axis=-1)

    res = {"gen_samples.uni": uni, "gen_samples.nor": nor}

    return res


def test_gen_samples():
    res = generic_gen_samples(pytest.model)

    for key, value in res.items():
        # % Check generate samples uniform/normal
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key
