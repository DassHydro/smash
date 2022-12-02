from __future__ import annotations

from smash.solver._mw_forward import forward_b

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model
    from smash.solver._mwd_parameters import ParametersDT
    from smash.solver._mwd_states import StatesDT

import numpy as np
from terminaltables import AsciiTable
import copy

__all__ = ["Net"]


class Net(object):
    """
    Artificial Neural Network initialization.

    TODO
    """

    def __init__(self):

        self.layers = []

        self.history = {"loss_train": [], "loss_valid": []}

        self.optimizer = None

        self.learning_rate = None

        self._compiled = False

    def __repr__(self):

        ret = []

        ret.append(AsciiTable([["Net summary"]]).table)

        if self._compiled and len(self.layers) > 0:

            ret.append(f"Input Shape: {self.layers[0].input_shape}")

            tab = [["Layer (type)", "Output Shape", "Param #"]]

            tot_params = 0
            trainable_params = 0

            for layer in self.layers:

                layer_name = layer.layer_name()

                n_params = layer.parameters()

                out_shape = layer.output_shape()

                tab.append([layer_name, str(out_shape), str(n_params)])

                tot_params += n_params

                if layer.trainable:
                    trainable_params += n_params

            ret.append(AsciiTable(tab).table)

            ret.append(f"Total params: {tot_params}")
            ret.append(f"Trainable params: {trainable_params}")
            ret.append(f"Non-trainable params: {tot_params - trainable_params}")

        else:

            ret.append("The network does not contain layers or has not been compiled yet")

        return "\n".join(ret)

    @property
    def layers(self):
        """
        List of layers of the network.

        TODO
        """

        return self._layers

    @layers.setter
    def layers(self, value):

        # TODO: add checktype
        self._layers = value

    @property
    def history(self):
        """
        Training history.

        TODO
        """

        return self._history

    @history.setter
    def history(self, value):

        # TODO: add checktype
        self._history = value

    def add(self, layer: Layer):
        """
        Add layers to the neural network.

        Parameters
        ----------
        layer : Layer
            TODO
        """

        # If this is not the first layer added then set the input shape
        # to the output shape of the last added layer
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())

        # Add layer to the network
        self.layers.append(layer)

    def compile(self, optimizer: str = "adam", learning_rate: float = 0.001):
        """
        Compile the network and set optimizer.

        Parameters
        ----------
        optimizer : str, default adam
            Optimizer algorithm. Should be one of

            - 'sgd'
            - 'adam'
            - 'adagrad'
            - 'rmsprop'

        learning_rate : float, default 0.001
            Learning rate that determines the step size of the optimization problem.
        """

        if len(self.layers) > 0:

            opt = OPTIMIZERS[optimizer.lower()](learning_rate=learning_rate)

            for layer in self.layers:

                if hasattr(layer, "initialize"):

                    layer.initialize(optimizer=opt)

            self._compiled = True
            self.optimizer = optimizer
            self.learning_rate = learning_rate

        else:
            raise ValueError("The network does not contain layers")

    def copy(self):
        """
        Make a deepcopy of the Net.

        Returns
        -------
        Net
            A copy of Net.
        """

        return copy.deepcopy(self)

    def set_trainable(self, trainable: list[bool]):
        """
        Method which enables to train or freeze the weights and biases of the network's layers.

        Parameters
        ----------
        trainable : list of bool
            List of booleans with a length of the total number of the network's layers.

            .. note::
                Activation and scaling functions do not have any weights and biases,
                so it is not important to set trainable weights at these layers.
        """

        if len(trainable) == len(self.layers):

            for i, layer in enumerate(self.layers):
                layer.trainable = trainable[i]

        else:
            raise ValueError(
                f"Inconsistent length between trainable ({len(trainable)}) and the number of layers ({len(self.layers)})"
            )

    def _fit(
        self,
        x_train: np.ndarray,
        instance: Model,
        control_vector: np.ndarray,
        mask: np.ndarray,
        parameters_bgd: ParametersDT,
        states_bgd: StatesDT,
        validation: float | None,  # TODO: add validation criteria
        epochs: int,
        early_stopping: bool,
        verbose: bool,
    ):

        if not self._compiled:
            raise ValueError(f"The network has not been compiled yet")

        loss_opt = 0  # only use for early stopping purpose

        # train model
        for epo in range(epochs):

            # Forward propogation
            y_pred = self._forward_pass(x_train)

            # Calculate the gradient of the loss function wrt y_pred
            loss_grad = _hcost_prime(
                y_pred, control_vector, mask, instance, parameters_bgd, states_bgd
            )

            # Compute loss
            loss = _hcost(instance)

            # Calculate the infinity norm of the projected gradient
            proj_g = _inf_norm(loss_grad)

            # early stopping
            if early_stopping:

                if loss_opt > loss or epo == 0:
                    loss_opt = loss

                    for layer in self.layers:

                        if hasattr(layer, "weight"):
                            layer._weight = np.copy(layer.weight)

                        if hasattr(layer, "bias"):
                            layer._bias = np.copy(layer.bias)

            # Backpropagation
            self._backward_pass(loss_grad=loss_grad)

            if verbose:
                ret = []

                ret.append(f"{' ' * 4}At epoch")
                ret.append("{:3}".format(epo + 1))
                ret.append("J =" + "{:10.6f}".format(loss))
                ret.append("|proj g| =" + "{:10.6f}".format(proj_g))

                print((" " * 4).join(ret))

            self.history["loss_train"].append(loss)

        if early_stopping:

            for layer in self.layers:

                if hasattr(layer, "weight"):
                    layer.weight = np.copy(layer._weight)

                if hasattr(layer, "bias"):
                    layer.bias = np.copy(layer._bias)

        print(f"{' ' * 4}STOP: TOTAL NO. OF EPOCH EXCEEDS LIMIT")

    def _forward_pass(self, x_train: np.ndarray, training: bool = True):

        layer_output = x_train

        for layer in self.layers:

            layer_output = layer._forward_pass(layer_output, training)

        return layer_output

    def _backward_pass(self, loss_grad: np.ndarray):

        for layer in reversed(self.layers):

            loss_grad = layer._backward_pass(loss_grad)

    def _predict(self, x_train: np.ndarray):

        preds = self._forward_pass(x_train, training=False)

        return preds


### LAYER ###


class Layer(object):
    def set_input_shape(self, shape: tuple):
        """Sets the shape that the layer expects of the input in the forward
        pass method"""
        self.input_shape = shape

    def layer_name(self):
        """The name of the layer. Used in model summary."""
        return self.__class__.__name__

    def parameters(self):
        """The number of trainable parameters used by the layer"""
        return 0

    def _forward_pass(self, x: np.ndarray, training: bool):
        """Propogates the signal forward in the network"""
        raise NotImplementedError()

    def _backward_pass(self, accum_grad: np.ndarray):
        """Propogates the accumulated gradient backwards in the network.
        If the has trainable weights then these weights are also tuned in this method.
        As input (accum_grad) it receives the gradient with respect to the output of the layer and
        returns the gradient with respect to the output of the previous layer."""
        raise NotImplementedError()

    def output_shape(self):
        """The shape of the output produced by forward_pass"""
        raise NotImplementedError()


class Activation(Layer):
    """A layer that applies an activation operation to the input.

    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """

    def __init__(self, name: str):
        self.activation_name = name
        self.activation_func = ACTIVATION_FUNC[name.lower()]()
        self.trainable = True

    def layer_name(self):
        return "Activation (%s)" % (self.activation_func.__class__.__name__)

    def _forward_pass(self, x: np.ndarray, training: bool = True):
        self.layer_input = x
        return self.activation_func(x)

    def _backward_pass(self, accum_grad: np.ndarray):
        return accum_grad * self.activation_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape


class Scale(Layer):
    """Scale function for outputs from the last layer w.r.t. parameters bounds."""

    def __init__(self, name: str, lower: np.ndarray, upper: np.ndarray):
        self.scale_name = name
        self.scale_func = SCALE_FUNC[name.lower()](lower, upper)
        self.trainable = True

    def layer_name(self):
        return "Scale (%s)" % (self.scale_func.__class__.__name__)

    def _forward_pass(self, x, training=True):
        self.layer_input = x
        return self.scale_func(x)

    def _backward_pass(self, accum_grad):
        return accum_grad * self.scale_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape


class Dense(Layer):
    """A fully-connected NN layer.
    Parameters:
    -----------
    neurons: int
        The number of neurons in the layer.
    input_shape: tuple
        The expected input shape of the layer. For dense layers a single digit specifying
        the number of features of the input. Must be specified if it is the first layer in
        the network.
    """

    def __init__(self, neurons: int, input_shape: tuple | None = None):

        self.layer_input = None
        self.input_shape = input_shape
        self.neurons = neurons
        self.trainable = True
        self.weight = None
        self.bias = None

    def initialize(self, optimizer: function):
        # Initialize weights and biases
        limit = 1 / np.sqrt(self.input_shape[0])

        self.weight = np.random.uniform(
            -limit, limit, (self.input_shape[0], self.neurons)
        )
        self.bias = np.zeros((1, self.neurons))

        # Set optimizer
        self.weight_opt = copy.copy(optimizer)

        self.bias_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.weight.shape) + np.prod(self.bias.shape)

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
        self.weight = self.weight_opt.update(self.weight, grad_w)
        self.bias = self.bias_opt.update(self.bias, grad_w0)

        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass
        accum_grad = accum_grad.dot(weight.T)
        return accum_grad

    def output_shape(self):
        return (self.neurons,)


### ACTIVATION FUNCTIONS ###


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class Softmax:
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)


class TanH:
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


ACTIVATION_FUNC = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "selu": SELU,
    "elu": ELU,
    "softmax": Softmax,
    "leaky_relu": LeakyReLU,
    "tanh": TanH,
    "softplus": SoftPlus,
}

### Scaling functions ###


class MinMaxScale:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __call__(self, x):
        return self.lower + x * (self.upper - self.lower)

    def gradient(self, x):
        return self.upper - self.lower


SCALE_FUNC = {
    "minmaxscale": MinMaxScale,
}


### OPTIMIZER ###


class StochasticGradientDescent:
    def __init__(self, learning_rate: float, momentum: float = 0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.w_updt = None

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.w_updt is None:
            self.w_updt = np.zeros(np.shape(w))
        # Use momentum if set
        self.w_updt = self.momentum * self.w_updt + (1 - self.momentum) * grad_wrt_w
        # Move against the gradient to minimize loss
        return w - self.learning_rate * self.w_updt


class Adam:
    def __init__(self, learning_rate: float, b1: float = 0.9, b2: float = 0.999):
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.m = None
        self.v = None
        # Decay rates
        self.b1 = b1
        self.b2 = b2

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray):
        # If not initialized
        if self.m is None:
            self.m = np.zeros(np.shape(grad_wrt_w))
            self.v = np.zeros(np.shape(grad_wrt_w))

        self.m = self.b1 * self.m + (1 - self.b1) * grad_wrt_w
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad_wrt_w, 2)

        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        self.w_updt = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

        return w - self.w_updt


class Adagrad:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.G = None  # Sum of squares of the gradients
        self.eps = 1e-8

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray):
        # If not initialized
        if self.G is None:
            self.G = np.zeros(np.shape(w))
        # Add the square of the gradient of the loss function at w
        self.G += np.power(grad_wrt_w, 2)
        # Adaptive gradient with higher learning rate for sparse data
        return w - self.learning_rate * grad_wrt_w / np.sqrt(self.G + self.eps)


class RMSprop:
    def __init__(self, learning_rate: float, rho: float = 0.9):
        self.learning_rate = learning_rate
        self.Eg = None  # Running average of the square gradients at w
        self.eps = 1e-8
        self.rho = rho

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray):
        # If not initialized
        if self.Eg is None:
            self.Eg = np.zeros(np.shape(grad_wrt_w))

        self.Eg = self.rho * self.Eg + (1 - self.rho) * np.power(grad_wrt_w, 2)

        # Divide the learning rate for a weight by a running average of the magnitudes of recent
        # gradients for that weight
        return w - self.learning_rate * grad_wrt_w / np.sqrt(self.Eg + self.eps)


OPTIMIZERS = {
    "sgd": StochasticGradientDescent,
    "adam": Adam,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}


### LOSS ###


def _hcost(instance: Model):

    return instance.output.cost


def _hcost_prime(
    y: np.ndarray,
    control_vector: np.ndarray,
    mask: np.ndarray,
    instance: Model,
    parameters_bgd: ParametersDT,
    states_bgd: StatesDT,
):

    #% Set parameters and states
    for i, name in enumerate(control_vector):

        if name in instance.setup._parameters_name:
            getattr(instance.parameters, name)[mask] = y[:, i]

        else:
            getattr(instance.states, name)[mask] = y[:, i]

    parameters_b = instance.parameters.copy()

    states_b = instance.states.copy()

    output_b = instance.output.copy()

    cost = np.float32(0)
    cost_b = np.float32(1)

    forward_b(
        instance.setup,
        instance.mesh,
        instance.input_data,
        instance.parameters,
        parameters_b,
        parameters_bgd,
        instance.states,
        states_b,
        states_bgd,
        instance.output,
        output_b,
        cost,
        cost_b,
    )

    grad = np.transpose(
        [
            getattr(parameters_b, name)
            if name in instance.setup._parameters_name
            else getattr(states_b, name)
            for name in control_vector
        ]
    )[mask]

    return grad


def _inf_norm(grad: np.ndarray):

    return np.amax(np.abs(grad))
