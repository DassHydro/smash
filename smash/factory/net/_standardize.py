from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from smash._constant import ACTIVATION_FUNCTION, ACTIVATION_FUNCTION_CLASS, WB_INITIALIZER

if TYPE_CHECKING:
    from smash.factory.net.net import Net
    from smash.util._typing import Any, AnyTuple, ListLike, Numeric

from smash.factory.samples._standardize import (
    _standardize_generate_samples_random_state,
)


def _standardize_add_dense_args(
    net: Net,
    neurons: Numeric,
    input_shape: Numeric | tuple | list | None,
    activation: str | None,
    kernel_initializer: str,
    bias_initializer: str,
) -> AnyTuple:
    neurons = _standardize_integer("neurons", neurons)
    input_shape = _standardize_add_dense_input_shape(net, input_shape)
    activation = _standardize_activation(activation)
    kernel_initializer = _standardize_initializer(kernel_initializer)
    bias_initializer = _standardize_initializer(bias_initializer)

    return neurons, input_shape, activation, kernel_initializer, bias_initializer


def _standardize_add_conv2d_args(
    net: Net,
    filters: Numeric,
    filter_shape: Numeric | tuple | list,
    input_shape: Numeric | tuple | list | None,
    activation: str | None,
    kernel_initializer: str,
    bias_initializer: str,
) -> AnyTuple:
    filters = _standardize_integer("filters", filters)
    filter_shape = _standardize_add_conv2d_filter_shape(filter_shape)
    input_shape = _standardize_add_conv2d_input_shape(net, input_shape)
    activation = _standardize_activation(activation)
    kernel_initializer = _standardize_initializer(kernel_initializer)
    bias_initializer = _standardize_initializer(bias_initializer)

    return filters, filter_shape, input_shape, activation, kernel_initializer, bias_initializer


def _standardize_add_scale_args(net: Net, bounds: ListLike) -> AnyTuple:
    if net.layers:
        try:
            bound_in = net.layers[-1]._activation_func.bound
        except Exception:
            raise ValueError(
                "Cannot apply scaling function to unbounded outputs from previous layer"
            ) from None

    else:
        raise ValueError("Scaling layer cannot be the first layer of the network")

    if isinstance(bounds, (tuple, list, np.ndarray)):
        if len(bounds) != net.layers[-1].output_shape()[-1]:
            raise ValueError(
                f"Inconsistent size between bounds argument and the last dimension of the output "
                f"in previous layer: {len(bounds)} != {net.layers[-1].output_shape()[-1]}"
            )

        for value in bounds:
            if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
                if value[0] >= value[1]:
                    raise ValueError(
                        f"Lower bound value {value[0]} is greater than or equal to upper bound {value[1]}"
                    )

            else:
                raise TypeError(
                    "Each element in bounds argument must be of ListLike type "
                    "(List, Tuple, np.ndarray) of size 2"
                )

    else:
        raise TypeError("bounds argument must be of ListLike type (List, Tuple, np.ndarray)")

    return np.array(bounds), bound_in


def _standardize_add_dropout_args(drop_rate: Numeric) -> float:
    if not isinstance(drop_rate, (int, float)):
        raise TypeError("drop_rate must be of Numeric type (int, float)")

    return drop_rate


def _standardize_set_trainable_args(net: Net, trainable: tuple | list) -> list:
    if isinstance(trainable, (tuple, list)):
        trainable = [bool(t) for t in trainable]

    else:
        raise TypeError("trainable argument must be a list of boolean values")

    if len(trainable) != len(net.layers):
        raise ValueError(
            f"Inconsistent size between trainable argument and the number of layers: "
            f"{len(trainable)} != {len(net.layers)}"
        )

    return trainable


def _standardize_set_weight_args(
    net: Net, value: list[Any] | None, random_state: int | None
) -> list[np.ndarray]:
    value = _standardize_set_weight_bias_value(net, value, "weight")
    random_state = _standardize_generate_samples_random_state(random_state)

    return value, random_state


def _standardize_set_bias_args(
    net: Net, value: list[Any] | None, random_state: int | None
) -> list[np.ndarray]:
    value = _standardize_set_weight_bias_value(net, value, "bias")
    random_state = _standardize_generate_samples_random_state(random_state)

    return value, random_state


def _standardize_forward_pass_args(net: Net, x: np.ndarray) -> np.ndarray:
    if not net.layers:
        raise ValueError("The graph of the neural network has not been set yet")

    if any(layer.weight is None for layer in net.layers if hasattr(layer, "weight")):
        raise ValueError("The neural network weight is not set yet")

    if any(layer.bias is None for layer in net.layers if hasattr(layer, "bias")):
        raise ValueError("The neural network bias is not set yet")

    return _standardize_forward_pass_x(net.layers[0].input_shape, x)


def _standardize_set_weight_bias_value(
    net: Net,
    value: list[Any] | None,
    kind: str,
) -> list[np.ndarray] | None:
    if not net.layers:
        raise ValueError("The graph of the neural network has not been set yet")

    else:
        trainable_layers = [layer for layer in net.layers if hasattr(layer, kind)]

    if value is None:
        pass

    elif isinstance(value, list):
        if len(value) != len(trainable_layers):
            raise ValueError(
                f"Inconsistent size between value argument and the number of trainable layers: "
                f"{len(value)} != {len(trainable_layers)}"
            )

        else:
            for i, arr in enumerate(value):
                kind_shape = getattr(trainable_layers[i], f"{kind}_shape")

                if isinstance(arr, (int, float)):
                    value[i] = np.full(kind_shape, arr)

                elif isinstance(arr, np.ndarray):
                    if arr.shape != kind_shape:
                        raise ValueError(
                            f"Invalid shape for value argument. Could not broadcast input array "
                            f"from shape {arr.shape} into shape {kind_shape}"
                        )
                else:
                    raise TypeError(
                        "Each element of value argument must be of Numeric type (int, float) or np.ndarray"
                    )

    else:
        raise TypeError("value argument must be a list of a same size with layers")

    return value


def _standardize_forward_pass_x(input_shape: tuple, x: np.ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        if x.shape != input_shape:
            raise ValueError(
                f"Invalid shape for input x. Could not broadcast input array "
                f"from shape {x.shape} into shape {input_shape}"
            )
    else:
        raise TypeError("Input x must be a np.ndarray")

    return x


def _standardize_add_conv2d_filter_shape(filter_shape: Numeric | tuple | list) -> tuple:
    if isinstance(filter_shape, (int, float)):
        filter_shape = (int(filter_shape),) * 2

    elif isinstance(filter_shape, (tuple, list)):
        filter_shape = tuple(filter_shape)

        if len(filter_shape) > 2:
            raise ValueError("filter_shape must be of size 2")

    else:
        raise TypeError("filter_shape must be an integer or a tuple")

    return filter_shape


def _standardize_add_dense_input_shape(net: Net, input_shape: Numeric | tuple | list | None) -> tuple:
    if net.layers:  # If this is not the first layer
        if input_shape is None:
            # Set the input shape to the output shape of the next added layer
            input_shape = net.layers[-1].output_shape()

        else:
            raise TypeError("input_shape argument should not be set for hidden and output layers")

    else:
        if input_shape is None:
            raise TypeError("First layer missing required argument: 'input_shape'")

        elif isinstance(input_shape, (int, float)):
            input_shape = (int(input_shape),)

        elif isinstance(input_shape, (tuple, list)):
            input_shape = tuple(input_shape)

        else:
            raise TypeError("input_shape argument must be an integer, a tuple, or a list")

    return input_shape


def _standardize_add_conv2d_input_shape(net: Net, input_shape: tuple | list | None) -> tuple:
    if net.layers:  # If this is not the first layer
        if input_shape is None:
            # Set the input shape to the output shape of the next added layer
            input_shape = net.layers[-1].output_shape()

        else:
            raise TypeError("input_shape argument should not be set for hidden and output layers")

    else:
        if input_shape is None:
            raise TypeError("First layer missing required argument: 'input_shape'")

        elif isinstance(input_shape, (tuple, list)):
            input_shape = tuple(input_shape)

            if len(input_shape) != 3:
                raise ValueError("input_shape of a Conv2D layer must be of size 3")

        else:
            raise TypeError("input_shape argument must be a tuple or a list")

    return input_shape


def _standardize_activation(activation: str | None) -> str | None:
    if activation is None:
        pass

    elif isinstance(activation, str):
        if activation.lower() in ACTIVATION_FUNCTION:
            ind = ACTIVATION_FUNCTION.index(activation.lower())
            activation = ACTIVATION_FUNCTION_CLASS[ind]

        else:
            raise ValueError(f"Unknown activation function {activation}. Choices: {ACTIVATION_FUNCTION}")

    else:
        raise TypeError("activation must be a str")

    return activation


def _standardize_initializer(initializer: str) -> str:
    if isinstance(initializer, str):
        if initializer.lower() not in WB_INITIALIZER:
            raise ValueError(f"Unknown initializer: {initializer}. Choices {WB_INITIALIZER}")
    else:
        raise TypeError("Initializer method must be a str")

    return initializer.lower()


def _standardize_integer(name: str, value: Numeric) -> int:
    if isinstance(value, (int, float)):
        value = int(value)

    else:
        raise TypeError(f"{name} argument must be of Numeric type (int, float)")

    return value
