from __future__ import annotations

import copy

import numpy as np

from smash._constant import WB_INITIALIZER


class Layer(object):
    def _set_input_shape(self, shape: tuple):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def n_params(self):
        return 0

    def _forward_pass(self, x: np.ndarray, training: bool):
        raise NotImplementedError()

    def _backward_pass(self, accum_grad: np.ndarray):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()


class Activation(Layer):
    """
    Activation layer that applies a specified activation function to the input.

    Options
    -------
    name : str
        The name of the activation function that will be used. Should be one of

        - 'relu' : Rectified Linear Unit
        - 'sigmoid' : Sigmoid
        - 'selu' : Scaled Exponential Linear Unit
        - 'elu' : Exponential Linear Unit
        - 'softmax' : Softmax
        - 'leaky_relu' : Leaky Rectified Linear Unit
        - 'tanh' : Hyperbolic Tangent
        - 'softplus' : Softplus
    """

    # TODO: Add function check_unknown_options
    def __init__(self, name: str, **unknown_options):
        self.input_shape = None

        self._activation_func = eval(name)()
        self.activation_name = self._activation_func.__class__.__name__
        self.trainable = True

    def layer_name(self):
        return f"Activation ({self.activation_name})"

    def _forward_pass(self, x: np.ndarray, training: bool = True):
        self.layer_input = x
        return self._activation_func(x)

    def _backward_pass(self, accum_grad: np.ndarray):
        return accum_grad * self._activation_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape


class Scale(Layer):
    """
    Scale layer that applies the min-max scaling function to the outputs.

    Options
    -------
    bounds : ListLike
        A sequence of ``(min, max)`` values that the outputs will be scaled to.
    """

    # TODO: Add function check_unknown_options
    def __init__(self, bounds: list | tuple | np.ndarray, bound_in: tuple, **unknown_options):
        self.input_shape = None

        self._scale_func = MinMaxScale(np.array(bounds), bound_in)

        self.scale_name = self._scale_func.__class__.__name__

        self.trainable = True

    def layer_name(self):
        return f"Scale ({self.scale_name})"

    def _forward_pass(self, x, training=True):
        self.layer_input = x
        return self._scale_func(x)

    def _backward_pass(self, accum_grad):
        return accum_grad * self._scale_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape


def _wb_initialization(layer: Layer, attr: str):
    fin = layer.input_shape[0]
    fout = layer.neurons

    if attr == "bias":
        initializer = layer.bias_initializer
        shape = (1, fout)

    else:
        initializer = layer.kernel_initializer
        shape = (fin, fout)

    split_inizer = initializer.split("_")

    if split_inizer[-1] == "uniform":
        if split_inizer[0] == "glorot":
            limit = np.sqrt(6 / (fin + fout))

        elif split_inizer[0] == "he":
            limit = np.sqrt(6 / fin)

        else:
            limit = 1 / np.sqrt(fin)

        setattr(layer, attr, np.random.uniform(-limit, limit, shape))

    elif split_inizer[-1] == "normal":
        if split_inizer[0] == "glorot":
            std = np.sqrt(2 / (fin + fout))

        elif split_inizer[0] == "he":
            std = np.sqrt(2 / fin)

        else:
            std = 0.01

        setattr(layer, attr, np.random.normal(0, std, shape))

    else:
        setattr(layer, attr, np.zeros(shape))


class Dense(Layer):
    """
    Fully-connected (dense) layer.

    Options
    -------
    neurons : int
        The number of neurons in the layer.

    input_shape : tuple or None, default None
        The expected input shape of the dense layer.
        It must be specified if this is the first layer in the network.

    kernel_initializer : str, default 'glorot_uniform'
        Weight initialization method. Should be one of 'uniform', 'glorot_uniform', 'he_uniform', 'normal',
        'glorot_normal', 'he_normal', 'zeros'.

    bias_initializer : str, default 'zeros'
        Bias initialization method. Should be one of 'uniform', 'glorot_uniform', 'he_uniform', 'normal',
        'glorot_normal', 'he_normal', 'zeros'.
    """

    # TODO: Add function check_unknown_options
    def __init__(
        self,
        neurons: int,
        input_shape: tuple | None = None,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        **unknown_options,
    ):
        self.layer_input = None

        self.input_shape = input_shape

        self.neurons = neurons

        self.trainable = True

        self.weight = None

        self.bias = None

        self.kernel_initializer = kernel_initializer.lower()

        if self.kernel_initializer not in WB_INITIALIZER:
            raise ValueError(
                f"Unknown kernel initializer: {self.kernel_initializer}. Choices {WB_INITIALIZER}"
            )

        self.bias_initializer = bias_initializer.lower()

        if self.bias_initializer not in WB_INITIALIZER:
            raise ValueError(f"Unknown bias initializer: {self.bias_initializer}. Choices {WB_INITIALIZER}")

    # TODO TYPE HINT: replace function by Callable
    def _initialize(self, optimizer: function):  # noqa: F821
        # Initialize weights and biases if not initialized
        if self.weight is None:
            _wb_initialization(self, "weight")
        if self.bias is None:
            _wb_initialization(self, "bias")

        # Set optimizer
        self._weight_opt = copy.copy(optimizer)
        self._bias_opt = copy.copy(optimizer)

    def n_params(self):
        return self.input_shape[0] * self.neurons + self.neurons

    def _forward_pass(self, x: np.ndarray, training: bool = True):
        if training:
            self.layer_input = x
        return x.dot(self.weight) + self.bias

    def _backward_pass(self, accum_grad: np.ndarray):
        # Save weights used during forwards pass
        weight = self.weight

        if self.trainable:
            # Calculate gradient w.r.t layer weights
            grad_w = self.layer_input.T.dot(accum_grad)
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)

            # Update the layer weights
            self.weight = self._weight_opt.update(self.weight, grad_w)
            self.bias = self._bias_opt.update(self.bias, grad_w0)

        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass
        accum_grad = accum_grad.dot(weight.T)
        return accum_grad

    def output_shape(self):
        return (self.neurons,)


class Dropout(Layer):
    """
    Dropout layer that randomly sets the output of the previous layer to zero with a specified probability.

    Options
    -------
    drop_rate : float
        The probability of setting a given output value to zero.
    """

    # TODO: Add function check_unknown_options
    def __init__(self, drop_rate: float, **unknown_options):
        self.drop_rate = drop_rate

        self._mask = None

        self.input_shape = None

        self.trainable = True

    def _forward_pass(self, x: np.ndarray, training: bool = True):
        c = 1 - self.drop_rate

        if training:
            self._mask = np.random.uniform(size=x.shape) > self.drop_rate
            c = self._mask

        return x * c

    def _backward_pass(self, accum_grad: np.ndarray):
        return accum_grad * self._mask

    def output_shape(self):
        return self.input_shape


### ACTIVATION FUNCTIONS ###


class Sigmoid:
    def __init__(self):
        self.bound = (0, 1)

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class Softmax:
    def __init__(self):
        self.bound = (0, 1)

    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)


class TanH:
    def __init__(self):
        self.bound = (-1, 1)

    def __call__(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)


class ReLU:
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)


class LeakyReLU:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)


class ELU:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, x):
        return np.where(x >= 0.0, 1, self.__call__(x) + self.alpha)


class SELU:
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def __call__(self, x):
        return self.scale * np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, x):
        return self.scale * np.where(x >= 0.0, 1, self.alpha * np.exp(x))


class SoftPlus:
    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        return 1 / (1 + np.exp(-x))


### SCALING FUNCTION ###


class MinMaxScale:
    def __init__(self, bounds: np.ndarray, bound_in: tuple):
        self._bounds = bounds

        self.lower = np.array([b[0] for b in bounds])
        self.upper = np.array([b[1] for b in bounds])

        self.bound_in_lower, self.bound_in_upper = bound_in

    def __call__(self, x: np.ndarray):
        return self.lower + (x - self.bound_in_lower) / (self.bound_in_upper - self.bound_in_lower) * (
            self.upper - self.lower
        )

    def gradient(self, x: np.ndarray):
        return (self.upper - self.lower) / (self.bound_in_upper - self.bound_in_lower)
