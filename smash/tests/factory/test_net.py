from __future__ import annotations

import numpy as np
import pytest

import smash


def generic_net_init(**kwargs):
    res = {}

    net = smash.factory.Net()

    n_hidden_layers = 4
    n_filters = 16

    for i in range(n_hidden_layers):
        if i == 0:
            net.add_conv2d(
                n_filters,
                filter_shape=(3, 4),
                input_shape=(12, 15, 2),
                kernel_initializer="he_normal",
                activation="leakyrelu",
            )
            net.add_flatten()

        else:
            n_neurons_i = round(n_filters * (n_hidden_layers - i) / n_hidden_layers)

            net.add_dense(n_neurons_i, kernel_initializer="he_uniform", activation="relu")
            net.add_dropout(0.1)

    net.add_dense(2, kernel_initializer="glorot_normal", activation="sigmoid")
    net.add_scale([(1.5, 3), (2, 5.5)])

    net._compile(
        optimizer="adam",
        learning_param={"learning_rate": 0.002},
        random_state=11,
    )

    graph = np.array([layer.layer_name() for layer in net.layers]).astype("S")

    res["net_init.graph"] = graph

    for i in range(n_hidden_layers):
        layer = net.layers[3 * i]

        res[f"net_init.weight_layer_{i+1}"] = layer.weight

        res[f"net_init.bias_layer_{i+1}"] = layer.bias

    return res


def generic_net_forward_pass(**kwargs):
    res = {}

    net = smash.factory.Net()

    # % Set NN graph
    net.add_conv2d(
        filters=64,
        filter_shape=(8, 6),
        input_shape=(12, 11, 3),
    )
    net.add_conv2d(
        filters=32,
        filter_shape=4,
        activation="selu",
    )
    net.add_flatten()
    net.add_dense(neurons=16, activation="elu")
    net.add_dense(neurons=4, activation="softplus")

    # % Set random weights
    net.set_weight(random_state=0)

    # % Set random biases
    np.random.seed(1)
    net.set_bias([0.01, 0, 0.02, np.random.uniform(size=(1, 4))])

    # % Forward pass
    np.random.seed(2)
    x = np.random.normal(0, 0.01, size=(12, 11, 3))
    y = net.forward_pass(x)

    res["net_forward_pass.output"] = y

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


def test_net_forward_pass():
    res = generic_net_forward_pass()

    key = "net_forward_pass.output"

    assert np.allclose(res[key], pytest.baseline[key][:], atol=1e-06), key
