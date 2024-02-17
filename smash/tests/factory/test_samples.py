from __future__ import annotations

import numpy as np
import pytest

import smash


def generic_generate_samples(model: smash.Model, **kwargs) -> dict:
    # % Fix problem
    num_vars = 4
    names = ["x1", "x2", "x3", "x4"]
    bounds = [(0, 77), (-940, 832), (23, 543), (-4, 681)]
    problem = {"num_vars": num_vars, "names": names, "bounds": bounds}

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

    res = {"generate_samples.uniform": uni, "generate_samples.normal": nor}

    return res


def test_generate_samples():
    res = generic_generate_samples(pytest.model)

    for key, value in res.items():
        # % Check generate samples uniform/normal
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key
