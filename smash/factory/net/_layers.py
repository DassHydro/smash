from __future__ import annotations

import copy

import numpy as np


class Layer(object):
    def _set_input_shape(self, shape: tuple):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def n_params(self):
        return 0

    def _forward_pass(self, x: np.ndarray):
        raise NotImplementedError()

    def _backward_pass(self, accum_grad: np.ndarray):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()


class Activation(Layer):
    def __init__(self, name: str):
        self.input_shape = None

        self._activation_func = eval(name)()
        self.activation_name = self._activation_func.__class__.__name__
        self.trainable = False

    def layer_name(self):
        return f"Activation ({self.activation_name})"

    def _forward_pass(self, x: np.ndarray):
        self.layer_input = x
        return self._activation_func(x)

    def _backward_pass(self, accum_grad: np.ndarray):
        return accum_grad * self._activation_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape


class Scale(Layer):
    def __init__(self, bounds: np.ndarray, bound_in: tuple):
        self.input_shape = None

        self._scale_func = MinMaxScale(bounds, bound_in)

        self.scale_name = self._scale_func.__class__.__name__

        self.trainable = False

    def layer_name(self):
        return f"Scale ({self.scale_name})"

    def _forward_pass(self, x):
        self.layer_input = x
        return self._scale_func(x)

    def _backward_pass(self, accum_grad):
        return accum_grad * self._scale_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape


def _initialize_nn_parameter(n_in: int, n_out: int, initializer: str) -> np.ndarray:
    split_inizer = initializer.split("_")

    if split_inizer[-1] == "uniform":
        if split_inizer[0] == "glorot":
            limit = np.sqrt(6 / (n_in + n_out))

        elif split_inizer[0] == "he":
            limit = np.sqrt(6 / n_in)

        else:
            limit = 1 / np.sqrt(n_in)

        value = np.random.uniform(-limit, limit, (n_out, n_in))

    elif split_inizer[-1] == "normal":
        if split_inizer[0] == "glorot":
            std = np.sqrt(2 / (n_in + n_out))

        elif split_inizer[0] == "he":
            std = np.sqrt(2 / n_in)

        else:
            std = 0.01

        value = np.random.normal(0, std, (n_out, n_in))

    else:
        value = np.zeros((n_out, n_in))

    return value


def _set_initialized_wb_to_layer(layer: Layer, kind: str):
    n_out, n_in = layer.weight_shape

    if kind == "bias":
        initializer = layer.bias_initializer
        value = _initialize_nn_parameter(1, n_out, initializer)
        value = value.T

    elif kind == "weight":
        initializer = layer.kernel_initializer
        value = _initialize_nn_parameter(n_in, n_out, initializer)

    else:  # Should be unreachable
        pass

    setattr(layer, kind, value)


class Dense(Layer):
    def __init__(
        self,
        neurons: int,
        input_shape: tuple,
        kernel_initializer: str,
        bias_initializer: str,
    ):
        self.layer_input = None

        self.input_shape = input_shape

        self.neurons = neurons

        self.trainable = True

        self.weight = None
        self.weight_shape = (neurons, input_shape[-1])

        self.bias = None
        self.bias_shape = (1, neurons)

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def _initialize(self, optimizer: callable):
        if self.weight is None:
            _set_initialized_wb_to_layer(self, "weight")
        if self.bias is None:
            _set_initialized_wb_to_layer(self, "bias")

        self._weight_opt = copy.copy(optimizer)
        self._bias_opt = copy.copy(optimizer)

    def n_params(self):
        return self.neurons * (self.input_shape[-1] + 1)

    def _forward_pass(self, x: np.ndarray):
        self.layer_input = x
        return x.dot(self.weight.T) + self.bias

    def _backward_pass(self, accum_grad: np.ndarray):
        weight = self.weight

        if self.trainable:
            # Compute gradients w.r.t. weights and biases
            grad_weight = accum_grad.T.dot(self.layer_input)
            grad_bias = np.sum(accum_grad, axis=0, keepdims=True)

            # Update weights and biases
            self.weight = self._weight_opt.update(self.weight, grad_weight)
            self.bias = self._bias_opt.update(self.bias, grad_bias)

        # Gradient propogated back to previous layer
        return accum_grad.dot(weight)

    def output_shape(self):
        return self.input_shape[:-1] + (self.neurons,)


class Conv2D(Layer):
    def __init__(
        self,
        filters: int,
        filter_shape: tuple,
        input_shape: tuple,
        kernel_initializer: str,
        bias_initializer: str,
    ):
        self.layer_input = None

        self.filters = filters
        self.filter_shape = filter_shape

        self.input_shape = input_shape

        self.trainable = True

        self.weight = None
        self.weight_shape = (filters, input_shape[-1] * np.prod(filter_shape))
        # The real W shape of a Conv2D layer is (filters, depth, height, width)
        # = (filters, input_shape[-1], **filter_shape),
        # which is reshaped as (filters, depth*height*width)

        self.bias = None
        self.bias_shape = (1, filters)

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def _initialize(self, optimizer: callable):
        if self.weight is None:
            _set_initialized_wb_to_layer(self, "weight")
        if self.bias is None:
            _set_initialized_wb_to_layer(self, "bias")

        self._weight_opt = copy.copy(optimizer)
        self._bias_opt = copy.copy(optimizer)

    def n_params(self):
        return self.filters * (self.input_shape[-1] * np.prod(self.filter_shape) + 1)

    def _forward_pass(self, x: np.ndarray):
        self.layer_input = x

        self.x_col = _im2col(x, self.filter_shape)

        res = self.x_col.dot(self.weight.T) + self.bias

        return res.reshape(self.output_shape())

    def _backward_pass(self, accum_grad):
        weight = self.weight

        # Reshape accum_grad into column shape
        accum_grad = accum_grad.reshape(-1, self.filters)

        if self.trainable:
            # Compute gradients w.r.t. weights and biases
            grad_weight = accum_grad.T.dot(self.x_col)
            grad_bias = np.sum(accum_grad, axis=0, keepdims=True)

            # Update weights and biases
            self.weight = self._weight_opt.update(self.weight, grad_weight)
            self.bias = self._bias_opt.update(self.bias, grad_bias)

        # Gradient propogated back to previous layer
        accum_grad = accum_grad.dot(weight)
        accum_grad = _col2im(accum_grad, self.input_shape, self.filter_shape)

        return accum_grad

    def output_shape(self):
        height, width, _ = self.input_shape

        return height, width, self.filters


class Flatten(Layer):
    def __init__(self):
        self._prev_shape = None

        self.input_shape = None

        self.trainable = False

    def _forward_pass(self, x):
        self._prev_shape = x.shape

        return x.reshape((-1, x.shape[-1]))

    def _backward_pass(self, accum_grad):
        return accum_grad.reshape(self._prev_shape)

    def output_shape(self):
        return (self.input_shape[0] * self.input_shape[1], self.input_shape[2])


class Dropout(Layer):
    def __init__(self, drop_rate: float):
        self.drop_rate = drop_rate

        self._mask = None

        self.input_shape = None

        self.trainable = False

    def _forward_pass(self, x: np.ndarray):
        c = 1 - self.drop_rate

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
        return 1 - 2 / (1 + np.exp(2 * x))

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)


class ReLU:
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)


class LeakyReLU:
    def __init__(self, alpha=0.01):
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


class SiLU:
    def __call__(self, x):
        return x / (1 + np.exp(-x))

    def gradient(self, x):
        s = 1 / (1 + np.exp(-x))
        return s + x * s * (1 - s)


### SCALING FUNCTION ###


class MinMaxScale:
    def __init__(self, bounds: np.ndarray, bound_in: tuple):
        self.lower = bounds[:, 0]
        self.upper = bounds[:, 1]

        self.bound_in_lower, self.bound_in_upper = bound_in

    def __call__(self, x: np.ndarray):
        return self.lower + (x - self.bound_in_lower) / (self.bound_in_upper - self.bound_in_lower) * (
            self.upper - self.lower
        )

    def gradient(self, x: np.ndarray):
        return (self.upper - self.lower) / (self.bound_in_upper - self.bound_in_lower)


### UTILS ###


def _im2col(im: np.ndarray, filter_shape: tuple) -> np.ndarray:
    pad_h, pad_w = _same_padding(filter_shape)

    im_padded = np.pad(im, (pad_h, pad_w, (0, 0)), mode="constant")

    i, j, k = _get_imcol_indices(im.shape, filter_shape, (pad_h, pad_w))

    return np.transpose(im_padded[i, j, k])


def _col2im(col: np.ndarray, im_shape: tuple, filter_shape: tuple) -> np.ndarray:
    height, width, depth = im_shape

    pad_h, pad_w = _same_padding(filter_shape)

    im_padded = np.zeros((height + np.sum(pad_h), width + np.sum(pad_w), depth))

    i, j, k = _get_imcol_indices(im_shape, filter_shape, (pad_h, pad_w))

    np.add.at(im_padded, (i, j, k), col.T)

    # Remove padding
    return im_padded[pad_h[0] : -pad_h[1], pad_w[0] : -pad_w[1]]


def _get_imcol_indices(im_shape: tuple, filter_shape: tuple, padding: tuple) -> tuple:
    height, width, depth = im_shape
    filter_height, filter_width = filter_shape

    pad_h, pad_w = padding

    height_out = height + np.sum(pad_h) - filter_height + 1
    width_out = width + np.sum(pad_w) - filter_width + 1

    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, depth)
    i1 = np.repeat(np.arange(height_out), width_out)

    j0 = np.tile(np.arange(filter_width), filter_height * depth)
    j1 = np.tile(np.arange(width_out), height_out)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(depth), filter_height * filter_width).reshape(-1, 1)

    return (i, j, k)


def _same_padding(filter_shape: tuple) -> tuple:
    # Compute 'same' padding where the output size is the same as input size

    pad_h1 = int((filter_shape[0] - 1) // 2)
    pad_h2 = filter_shape[0] - 1 - pad_h1
    pad_w1 = int((filter_shape[1] - 1) // 2)
    pad_w2 = filter_shape[1] - 1 - pad_w1

    return (pad_h1, pad_h2), (pad_w1, pad_w2)
