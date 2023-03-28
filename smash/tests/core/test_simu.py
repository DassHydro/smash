from __future__ import annotations

import smash

from smash.solver._mwd_cost import nse, kge
from smash.core._constant import MAPPING

import numpy as np
import pytest


def generic_run(model: smash.Model, **kwargs) -> dict:
    instance = model.run()

    res = {"run.cost": output_cost(instance)}

    return res


def test_run():
    res = generic_run(pytest.model)

    for key, value in res.items():
        # % Check cost in run
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key


def generic_optimize(model: smash.Model, **kwargs) -> dict:
    res = {}

    for mapping in MAPPING:
        if mapping == "uniform":
            algo = "sbs"

        else:
            algo = "l-bfgs-b"

        instance = model.optimize(
            mapping=mapping,
            algorithm=algo,
            options={"maxiter": 1},
            verbose=False,
        )

        res[f"optimize.{mapping}_{algo}.cost"] = output_cost(instance)

    # % multi-criteria
    instance = model.optimize(
        mapping="uniform",
        algorithm="nelder-mead",
        jobs_fun=["nse", "Crc", "Cfp10", "Cfp50", "Epf", "Elt"],
        wjobs_fun=[1, 2, 2, 2, 2, 2],
        event_seg={"peak_quant": 0.99},
        options={"maxiter": 10},
        verbose=False,
    )

    res["optimize.uniform_nelder-mead.cost"] = output_cost(instance)

    # % multi-gauges
    instance = model.optimize(
        gauge="all",
        wgauge="median",
        options={"maxiter": 1},
        verbose=False,
    )

    res["optimize.uniform_sbs_mtg.cost"] = output_cost(instance)

    # % states hp optimization
    tmp = model.states.hlr.copy()
    model.states.hlr = 1
    instance = model.optimize(
        control_vector=["cp", "hlr"], options={"maxiter": 1}, verbose=False
    )
    model.states.hlr = tmp.copy()

    res["optimize.uniform_sbs_states.cost"] = output_cost(instance)
    res["optimize.uniform_sbs_states.hlr"] = instance.states.hlr.copy()

    # % adjust bounds
    instance = model.optimize(
        mapping="distributed",
        algorithm="l-bfgs-b",
        control_vector=["cp", "cft"],
        bounds={"cp": [1, 300]},
        options={"maxiter": 1},
        verbose=False,
    )

    res["optimize.distributed_l-bfgs-b_bounds.cost"] = output_cost(instance)
    res["optimize.distributed_l-bfgs-b_bounds.cp"] = instance.parameters.cp.copy()
    res["optimize.distributed_l-bfgs-b_bounds.cft"] = instance.parameters.cft.copy()

    return res


def test_optimize():
    res = generic_optimize(pytest.model)

    for key, value in res.items():
        # % Check cost in optimize
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key

    # % Check that optimize sbs does not modify parameters that are not optimized
    # % (case of spatially distributed prior only)
    # % Does not need a baseline. Should work like this in any case
    tmp = pytest.model.parameters.cft.copy()
    pytest.model.parameters.cft = np.random.rand(*tmp.shape) + 500

    instance = pytest.model.optimize(
        control_vector="cp", options={"maxiter": 1}, verbose=False
    )

    assert np.array_equal(
        pytest.model.parameters.cft, instance.parameters.cft
    ), "optimize.uniform_sbs_sdp.cft"

    pytest.model.parameters.cft = tmp.copy()


def generic_bayes_estimate(model: smash.Model, **kwargs) -> dict:
    instance, br = model.bayes_estimate(
        k=np.linspace(-1, 5, 10),
        n=5,
        return_br=True,
        random_state=11,
        verbose=False,
    )

    res = {
        "bayes_estimate.br_cost": np.array(br.l_curve["cost"]),
        "bayes_estimate.cost": output_cost(instance),
    }

    return res


def test_bayes_estimate():
    res = generic_bayes_estimate(pytest.model)

    for key, value in res.items():
        # % Check br.l_curve and cost in bayes_estimate
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key


def generic_bayes_optimize(model: smash.Model, **kwargs) -> dict:
    instance, br = model.bayes_optimize(
        k=np.linspace(-1, 5, 10),
        n=5,
        mapping="distributed",
        algorithm="l-bfgs-b",
        options={"maxiter": 1},
        return_br=True,
        random_state=11,
        verbose=False,
    )

    res = {
        "bayes_optimize.br_cost": np.array(br.l_curve["cost"]),
        "bayes_optimize.cost": output_cost(instance),
    }

    return res


def test_bayes_optimize():
    res = generic_bayes_optimize(pytest.model)

    for key, value in res.items():
        # % Check br.l_curve and cost in bayes_optimize
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key


def generic_ann_optimize_1(model: smash.Model, **kwargs) -> dict:
    instance, net = model.ann_optimize(
        epochs=5, learning_rate=0.001, return_net=True, random_state=11, verbose=False
    )

    res = {
        "ann_optimize_1.loss": np.array(net.history["loss_train"]),
        "ann_optimize_1.cost": output_cost(instance),
    }

    return res


def test_ann_optimize_1():
    res = generic_ann_optimize_1(pytest.model)

    for key, value in res.items():
        # % Check net.history loss and cost in ann_optimize_1
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key


def generic_ann_optimize_2(model: smash.Model, **kwargs) -> dict:
    problem = model.get_bound_constraints(states=False)

    nd = model.input_data.descriptor.shape[-1]
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
        options={"learning_rate": 0.01, "momentum": 0.001},
        random_state=11,
    )

    instance = model.ann_optimize(net=net, epochs=5, verbose=False)

    res = {
        "ann_optimize_2.loss": np.array(net.history["loss_train"]),
        "ann_optimize_2.cost": output_cost(instance),
    }

    return res


def test_ann_optimize_2():
    res = generic_ann_optimize_2(pytest.model)

    for key, value in res.items():
        # % Check net.history loss and cost in ann_optimize_2
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key


def output_cost(instance: smash.Model):
    qo = instance.input_data.qobs
    qs = instance.output.qsim

    n_jtest = 3

    ret = np.zeros(shape=n_jtest * instance.mesh.ng, dtype=np.float32)

    for i in range(instance.mesh.ng):
        ret[n_jtest * i : n_jtest * (i + 1)] = (
            instance.output.cost,
            nse(qo[i], qs[i]),
            kge(qo[i], qs[i]),
        )

    return np.array(ret)
