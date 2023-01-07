import smash
import numpy as np
import pytest


def test_gen_samples():

    problem = pytest.model.get_bound_constraints()

    sample_uni = smash.generate_samples(problem, generator="uniform", n=20, random_state=11).to_numpy()

    sample_nor = smash.generate_samples(problem, generator="normal", n=20, random_state=11).to_numpy()

    assert np.allclose(sample_uni, pytest.baseline["gen_samples.uni"][:], atol=1e-06)

    assert np.allclose(sample_nor, pytest.baseline["gen_samples.nor"][:], atol=1e-06)



