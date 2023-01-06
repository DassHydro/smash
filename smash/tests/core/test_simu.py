from __future__ import annotations

import smash

from smash.solver._mwd_cost import nse, kge
from smash.core._constant import ALGORITHM

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

import numpy as np
import pytest


def test_direct_run():

    instance = pytest.model.copy()

    instance.run(inplace=True)

    assert np.allclose(
        output_cost(instance), pytest.baseline["simu.run"][:], atol=1e-06
    )


def test_optimize():

    for algo in ALGORITHM:

        instance = pytest.model.copy()

        if algo == "l-bfgs-b":
            mapping = "distributed"

        else:
            mapping = "uniform"

        instance.optimize(
            mapping=mapping,
            algorithm=algo,
            options={"maxiter": 2},
            inplace=True,
            verbose=False,
        )

        assert np.allclose(
            output_cost(instance), pytest.baseline[f"simu.{algo}"][:], atol=1e-06
        )


# TODO: add more tests for model.optimize


def test_bayes_estimate():

    instance = pytest.model.copy()

    br = instance.bayes_estimate(
        k=np.linspace(-1, 5, 20),
        n=10,
        inplace=True,
        return_br=True,
        random_state=11,
        verbose=False,
    )

    assert np.allclose(
        br.l_curve["cost"], pytest.baseline["simu.bayes_estimate_br_cost"], atol=1e-06
    )

    assert np.allclose(
        output_cost(instance), pytest.baseline["simu.bayes_estimate"][:], atol=1e-06
    )


def test_bayes_optimize():

    instance = pytest.model.copy()

    br = instance.bayes_optimize(
        k=np.linspace(-1, 5, 20),
        n=5,
        mapping="distributed",
        algorithm="l-bfgs-b",
        options={"maxiter": 1},
        inplace=True,
        return_br=True,
        random_state=11,
        verbose=False,
    )

    assert np.allclose(
        br.l_curve["cost"],
        pytest.baseline["simu.bayes_optimize_br_cost"][:],
        atol=1e-06,
    )

    assert np.allclose(
        output_cost(instance), pytest.baseline["simu.bayes_optimize"][:], atol=1e-06
    )


def test_ann_optimize_1():

    instance = pytest.model.copy()

    np.random.seed(11)
    net = instance.ann_optimize(epochs=10, inplace=True, return_net=True, verbose=False)

    assert np.allclose(
        net.history["loss_train"],
        pytest.baseline["simu.ann_optimize_1_loss"][:],
        atol=1e-06,
    )

    assert np.allclose(
        output_cost(instance), pytest.baseline["simu.ann_optimize_1"][:], atol=1e-06
    )


def test_ann_optimize_2():

    instance = pytest.model.copy()

    problem = instance.get_bound_constraints(states=False)

    nd = instance.input_data.descriptor.shape[-1]
    ncv = problem["num_vars"]
    bounds = problem["bounds"]

    net = smash.Net()

    net.add(layer="dense", options={"input_shape": (nd,), "neurons": 16})
    net.add(layer="activation", options={"name": "relu"})

    net.add(layer="dense", options={"neurons": 8})
    net.add(layer="activation", options={"name": "relu"})

    net.add(layer="dense", options={"neurons": ncv})
    net.add(layer="activation", options={"name": "sigmoid"})

    net.add(layer="scale", options={"bounds": bounds})

    net.compile(
        optimizer="sgd",
        learning_rate=0.01,
        options={"momentum": 0.001},
        random_state=11,
    )

    instance.ann_optimize(net=net, epochs=10, inplace=True, verbose=False)

    assert np.allclose(
        net.history["loss_train"],
        pytest.baseline["simu.ann_optimize_2_loss"][:],
        atol=1e-06,
    )

    assert np.allclose(
        output_cost(instance), pytest.baseline["simu.ann_optimize_2"][:], atol=1e-06
    )


def output_cost(instance: Model):

    qo = instance.input_data.qobs
    qs = instance.output.qsim

    ret = []

    for i in range(instance.mesh.code.size):

        ret.append(nse(qo[i], qs[i]))
        ret.append(kge(qo[i], qs[i]))

    return np.array(ret)
