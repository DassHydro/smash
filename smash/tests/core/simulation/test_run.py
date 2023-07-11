from __future__ import annotations

import smash

from smash.fcore._mwd_efficiency_metric import nse, kge

import numpy as np
import pytest


def generic_run(model: smash.Model, **kwargs) -> dict:
    instance = smash.forward_run(model)

    res = {"run.cost": output_cost(instance)}

    return res


def test_run():
    res = generic_run(pytest.model)

    for key, value in res.items():
        # % Check cost in run
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key


def output_cost(instance: smash.Model):
    qo = instance.obs_response.q
    qs = instance.sim_response.q

    n_jtest = 3

    ret = np.zeros(shape=n_jtest * instance.mesh.ng, dtype=np.float32)

    for i in range(instance.mesh.ng):
        ret[n_jtest * i : n_jtest * (i + 1)] = (
            0,
            nse(qo[i], qs[i]),
            kge(qo[i], qs[i]),
        )

    return np.array(ret)
