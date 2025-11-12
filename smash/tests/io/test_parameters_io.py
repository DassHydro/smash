from __future__ import annotations

import numpy as np
import pytest

import smash


def test_parameters_io():
    model_tmp = pytest.model.copy()
    smash.io.save_grid_parameters(model_tmp, "tmp_parameters_io")
    smash.io.read_grid_parameters(model_tmp, "tmp_parameters_io")
    for key in pytest.model.rr_parameters.keys:
        assert np.all(pytest.model.get_rr_parameters(key) == model_tmp.get_rr_parameters(key)), (
            f"parameters_io.{key}"
        )

    smash.io.save_grid_parameters(model_tmp, "tmp_parameters_io", parameters=model_tmp.rr_initial_states.keys)
    smash.io.read_grid_parameters(model_tmp, "tmp_parameters_io", parameters=model_tmp.rr_initial_states.keys)
    for key in pytest.model.rr_initial_states.keys:
        assert np.all(pytest.model.get_rr_initial_states(key) == model_tmp.get_rr_initial_states(key)), (
            f"parameters_io.{key}"
        )
