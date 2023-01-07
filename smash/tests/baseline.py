from __future__ import annotations

import smash

from smash.core._constant import ALGORITHM, STRUCTURE_PARAMETERS, CSIGN, ESIGN

from core.test_simu import output_cost

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

import os

import numpy as np
from h5py import File


def baseline_simu(f: File, model: Model):

    ### direct run
    instance = model.copy()

    instance.run(inplace=True)

    res = output_cost(instance)

    f.create_dataset("simu.run", data=res)

    del instance

    ### optimize
    for algo in ALGORITHM:

        instance = model.copy()

        if algo == "l-bfgs-b":
            mapping = "distributed"

        else:
            mapping = "uniform"

        instance.optimize(
            mapping=mapping, algorithm=algo, options={"maxiter": 1}, inplace=True
        )

        res = output_cost(instance)

        f.create_dataset(f"simu.{algo}", data=res)

        del instance

    ### bayes_estimate
    instance = model.copy()

    br = instance.bayes_estimate(
        k=np.linspace(-1, 5, 10),
        n=5,
        inplace=True,
        return_br=True,
        random_state=11,
    )

    f.create_dataset("simu.bayes_estimate_br_cost", data=br.l_curve["cost"])

    res = output_cost(instance)

    f.create_dataset("simu.bayes_estimate", data=res)

    del instance

    ### bayes_optimize
    instance = model.copy()

    br = instance.bayes_optimize(
        k=np.linspace(-1, 5, 10),
        n=5,
        mapping="distributed",
        algorithm="l-bfgs-b",
        options={"maxiter": 1},
        inplace=True,
        return_br=True,
        random_state=11,
    )

    f.create_dataset("simu.bayes_optimize_br_cost", data=br.l_curve["cost"])

    res = output_cost(instance)

    f.create_dataset("simu.bayes_optimize", data=res)

    del instance

    ### ann_optimize with default graph
    instance = model.copy()

    np.random.seed(11)
    net = instance.ann_optimize(epochs=5, inplace=True, return_net=True)

    f.create_dataset("simu.ann_optimize_1_loss", data=net.history["loss_train"])

    res = output_cost(instance)

    f.create_dataset("simu.ann_optimize_1", data=res)

    del instance

    ### ann_optimize with redefined graph
    instance = model.copy()

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

    instance.ann_optimize(net=net, epochs=5, inplace=True)

    f.create_dataset("simu.ann_optimize_2_loss", data=net.history["loss_train"])

    res = output_cost(instance)

    f.create_dataset("simu.ann_optimize_2", data=res)

    del instance


def baseline_signatures(f: File, model: Model):

    instance = model.copy()
    instance.run(inplace=True)

    ### signatures computation
    signresult = instance.signatures()

    for typ, sign in zip(
        ["cont", "event"], [CSIGN[:4], ESIGN]
    ):  # remove percentile signatures calculation

        for dom in ["obs", "sim"]:

            arr = signresult[typ][dom][sign].to_numpy(dtype=np.float32)

            f.create_dataset(
                f"signatures.{typ}_{dom}",
                shape=arr.shape,
                dtype=arr.dtype,
                data=arr,
                compression="gzip",
                chunks=True,
            )

    ### signatures sensitivity
    signsensresult = model.signatures_sensitivity(n=8, random_state=11)
    for typ, sign in zip(
        ["cont", "event"], [CSIGN[:4], ESIGN]
    ):  # remove percentile signatures calculation

        for ordr in ["first_si", "total_si"]:

            for param in STRUCTURE_PARAMETERS[model.setup.structure]:

                arr = signsensresult[typ][ordr][param][sign].to_numpy(dtype=np.float32)

                f.create_dataset(
                    f"signatures_sens.{typ}_{ordr}_{param}",
                    shape=arr.shape,
                    dtype=arr.dtype,
                    data=arr,
                    compression="gzip",
                    chunks=True,
                )


def baseline_event_seg(f: File, model: Model):

    arr = model.event_segmentation().to_numpy()

    arr = arr.astype("S")

    f.create_dataset(
        "event_seg",
        shape=arr.shape,
        dtype=arr.dtype,
        data=arr,
        compression="gzip",
        chunks=True,
    )


def baseline_gen_samples(f: File, model: Model):

    problem = model.get_bound_constraints()

    sample_uni = smash.generate_samples(problem, generator="uniform", n=20, random_state=11).to_numpy()

    sample_nor = smash.generate_samples(problem, generator="normal", n=20, random_state=11).to_numpy()

    f.create_dataset(
                    "gen_samples.uni",
                    shape=sample_uni.shape,
                    dtype=sample_uni.dtype,
                    data=sample_uni,
                    compression="gzip",
                    chunks=True,
                )

    f.create_dataset(
                    "gen_samples.nor",
                    shape=sample_nor.shape,
                    dtype=sample_nor.dtype,
                    data=sample_nor,
                    compression="gzip",
                    chunks=True,
                )

def baseline_net(f: File, model: Model):
    
    net = smash.Net()

    n_hidden_layers = 4
    n_neurons = 16

    for i in range(n_hidden_layers):

        if i == 0:

            net.add(
                layer="dense",
                options={
                    "input_shape": (6,),
                    "neurons": n_neurons,
                    "kernel_initializer": "he_uniform",
                },
            )

        else:

            n_neurons_i = round(n_neurons * (n_hidden_layers - i)/n_hidden_layers)
            
            net.add(
                layer="dense",
                options={
                    "neurons": n_neurons_i,
                    "kernel_initializer": "he_uniform",
                },
            )

        net.add(layer="activation", options={"name": "relu"})
        net.add(layer="dropout", options={"drop_rate": .1})

    net.add(
        layer="dense",
        options={"neurons": 2, "kernel_initializer": "glorot_uniform"},
    )
    net.add(layer="activation", options={"name": "sigmoid"})

    net.compile("adam", learning_rate=0.002, options={"b1": 0.8, "b2": 0.99}, random_state=11)

    graph = np.array([l.layer_name() for l in net.layers]).astype("S")

    f.create_dataset(
        "net.graph",
        shape=graph.shape,
        dtype=graph.dtype,
        data=graph,
        compression="gzip",
        chunks=True,
    )

    for i in range(n_hidden_layers):

        layer = net.layers[3*i]

        f.create_dataset(
                    f"net.init_weight_layer_{i+1}",
                    shape=layer.weight.shape,
                    dtype=layer.weight.dtype,
                    data=layer.weight,
                    compression="gzip",
                    chunks=True,
                )

        f.create_dataset(
                    f"net.init_bias_layer_{i+1}",
                    shape=layer.bias.shape,
                    dtype=layer.bias.dtype,
                    data=layer.bias,
                    compression="gzip",
                    chunks=True,
                )


if __name__ == "__main__":

    setup, mesh = smash.load_dataset("cance")
    model = smash.Model(setup, mesh)

    if os.path.exists("baseline.hdf5"):
        os.remove("baseline.hdf5")

    with File("baseline.hdf5", "a") as f:

        baseline_simu(f, model)
        baseline_signatures(f, model)
        baseline_event_seg(f, model)
        baseline_gen_samples(f, model)
        baseline_net(f, model)

