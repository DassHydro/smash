import smash
import numpy as np
from h5py import File
import pytest
import os

from smash.core._constant import ALGORITHM

from baseline_simu import output_cost


setup, mesh = smash.load_dataset("cance")
pytest.model = smash.Model(setup, mesh)

pytest.baseline = File(os.path.join(os.path.dirname(__file__), "baseline_simu.hdf5"))


def test_direct_run():

    instance = pytest.model.copy()

    instance.run(inplace=True)

    assert np.array_equal(output_cost(instance), pytest.baseline["direct_run"])


def test_optimize():

    for algo in ALGORITHM:

        instance = pytest.model.copy()

        if algo=="l-bfgs-b":
            mapping = "distributed" 

        else:
            mapping = "uniform"

        instance.optimize(mapping=mapping, algorithm=algo, options={"maxiter": 2}, inplace=True, verbose=False)

        assert np.array_equal(output_cost(instance), pytest.baseline[algo])


# TODO: add more tests for model.optimize


def test_bayes_estimate():

    instance = pytest.model.copy()

    br = instance.bayes_estimate(k=np.linspace(-1, 5, 20), n=10, inplace=True, return_br=True, random_state=11, verbose=False)

    crit1 = np.array_equal(br.l_curve["cost"], pytest.baseline["bayes_estimate_br_cost"])

    crit2 = np.array_equal(output_cost(instance), pytest.baseline["bayes_estimate"])

    assert all([crit1, crit2])


def test_bayes_optimize():

    instance = pytest.model.copy()

    br = instance.bayes_optimize(k=np.linspace(-1, 5, 20), n=5, mapping="distributed", algorithm="l-bfgs-b", options={"maxiter": 1}, inplace=True, return_br=True, random_state=11, verbose=False)

    crit1 = np.array_equal(br.l_curve["cost"], pytest.baseline["bayes_optimize_br_cost"])

    crit2 = np.array_equal(output_cost(instance), pytest.baseline["bayes_optimize"])

    assert all([crit1, crit2])


def test_ann_optimize_1():

    instance = pytest.model.copy()

    np.random.seed(11)
    net = instance.ann_optimize(epochs=10, inplace=True, return_net=True, verbose=False)

    crit1 = np.array_equal(net.history["loss_train"], pytest.baseline["ann_optimize_1_loss"])

    crit2 = np.array_equal(output_cost(instance), pytest.baseline["ann_optimize_1"])

    assert all([crit1, crit2])


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

    net.compile(optimizer="sgd", learning_rate=0.01, options={'momentum': 0.001}, random_state=11)

    instance.ann_optimize(net=net, epochs=10, inplace=True, verbose=False)

    crit1 = np.array_equal(net.history["loss_train"], pytest.baseline["ann_optimize_2_loss"])

    crit2 = np.array_equal(output_cost(instance), pytest.baseline["ann_optimize_2"])

    assert all([crit1, crit2])
