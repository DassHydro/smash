import smash
import numpy as np
import pytest


def test_net_init():

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

    assert np.array_equal(graph, pytest.baseline["net.graph"])

    for i in range(n_hidden_layers):

        layer = net.layers[3*i]

        assert np.allclose(layer.weight, pytest.baseline[f"net.init_weight_layer_{i+1}"][:], atol=1e-06)

        assert np.allclose(layer.bias, pytest.baseline[f"net.init_bias_layer_{i+1}"][:], atol=1e-06)


# TODO add more tests (in addition to net init) if need


