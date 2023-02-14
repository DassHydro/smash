from __future__ import annotations

import smash

import numpy as np
import pytest


def generic_gen_samples(model: smash.Model, **kwargs) -> dict:
    problem = model.get_bound_constraints()

    uni = smash.generate_samples(
        problem, generator="uniform", n=20, random_state=11
    ).to_numpy()

    nor = smash.generate_samples(
        problem, generator="normal", n=20, random_state=11
    ).to_numpy()

    res = {"gen_samples.uni": uni, "gen_samples.nor": nor}

    return res


def test_gen_samples():
    res = generic_gen_samples(pytest.model)

    for key, value in res.items():
        # % Check generate samples uniform/normal
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key
