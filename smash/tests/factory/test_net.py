from __future__ import annotations

import numpy as np
import pytest

import smash


def generic_net_init(**kwargs):
    res = {}

    net = smash.factory.Net()

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
            n_neurons_i = round(n_neurons * (n_hidden_layers - i) / n_hidden_layers)

            net.add(
                layer="dense",
                options={
                    "neurons": n_neurons_i,
                    "kernel_initializer": "he_uniform",
                },
            )

        net.add(layer="activation", options={"name": "relu"})
        net.add(layer="dropout", options={"drop_rate": 0.1})

    net.add(
        layer="dense",
        options={"neurons": 2, "kernel_initializer": "glorot_uniform"},
    )
    net.add(layer="activation", options={"name": "sigmoid"})

    net._compile(
        optimizer="adam",
        learning_param={"learning_rate": 0.002, "b1": 0.8, "b2": 0.99},
        random_state=11,
    )  # % TODO: change this when net.fit_xx available

    graph = np.array([layer.layer_name() for layer in net.layers]).astype("S")

    res["net_init.graph"] = graph

    for i in range(n_hidden_layers):
        layer = net.layers[3 * i]

        res[f"net_init.weight_layer_{i+1}"] = layer.weight

        res[f"net_init.bias_layer_{i+1}"] = layer.bias

    return res


def test_net_init():
    res = generic_net_init()

    for key, value in res.items():
        if key in ["net_init.graph"]:
            # % Check net init graph
            assert np.array_equal(value, pytest.baseline[key][:]), key

        else:
            # % Check net init layer weight and bias
            assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key
