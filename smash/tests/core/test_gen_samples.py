from __future__ import annotations

import smash

import numpy as np
import pytest


def generic_gen_samples(model: smash.Model, **kwargs) -> dict:
    problem = model.get_bound_constraints()

    sample = smash.generate_samples(problem, generator="uniform", n=20, random_state=11)
    uni = sample.to_numpy(axis=-1)

    sample = smash.generate_samples(
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
