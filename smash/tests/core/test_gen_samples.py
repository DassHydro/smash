from __future__ import annotations

import smash

import numpy as np
import pandas as pd
import pytest


def generic_gen_samples(model: smash.Model, **kwargs) -> dict:
    problem = model.get_bound_constraints()

    sample = smash.generate_samples(problem, generator="uniform", n=20, random_state=11)
    uni = pd.DataFrame({key: sample[key] for key in problem["names"]}).to_numpy()

    sample = smash.generate_samples(
        problem,
        generator="normal",
        n=20,
        mean={problem["names"][1]: np.mean(problem["bounds"][1])},
        coef_std=3,
        random_state=11,
    )
    nor = pd.DataFrame({key: sample[key] for key in problem["names"]}).to_numpy()

    res = {"gen_samples.uni": uni, "gen_samples.nor": nor}

    return res


def test_gen_samples():
    res = generic_gen_samples(pytest.model)

    for key, value in res.items():
        # % Check generate samples uniform/normal
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key
