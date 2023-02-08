from __future__ import annotations

from smash.solver._mw_forward import forward_b

from smash.core._constant import WB_INITIALIZER

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model
    from smash.solver._mwd_parameters import ParametersDT
    from smash.solver._mwd_states import StatesDT

import warnings
import copy
import numpy as np
from terminaltables import AsciiTable
from tqdm import tqdm

__all__ = ["Net"]


class Net(object):
    """
    Artificial Neural Network initialization.

    Examples
    --------
    >>> net = smash.Net()
    >>> net
    +-------------+
    | Net summary |
    +-------------+
    The network does not contain layers or has not been compiled yet
    """

    def __init__(self):

        self.layers = []

        self.history = {"loss_train": [], "loss_valid": []}

        self._optimizer = None

        self._learning_rate = None

        self._compiled = False

    def __repr__(self):

        ret = []

        ret.append(AsciiTable([["Net summary"]]).table)

        if self._compiled and self.layers:

            ret.append(f"Input Shape: {self.layers[0].input_shape}")

            tab = [["Layer (type)", "Output Shape", "Param #"]]

            tot_params = 0
            trainable_params = 0

            for layer in self.layers:

                layer_name = layer.layer_name()

                n_params = layer.n_params()

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

            ret.append(
                "The network does not contain layers or has not been compiled yet"
            )

        return "\n".join(ret)

    @property
    def layers(self):
        """
        List of Layer objects defining the graph of the network.

        The graph is set using `smash.Net.add` method.

        Examples
        --------
        >>> net = smash.Net()
        >>> net.add(layer="dense", options={"input_shape": (6,), "neurons": 32})
        >>> net.add(layer="activation", options={"name": "sigmoid"})
        >>> net.add(layer="dropout", options={"drop_rate": .2})
        >>> net.compile()

        If you are using IPython, tab completion allows you to visualize all the attributes and methods of each Layer object:

        >>> layer_1 = net.layers[0]
        >>> layer_1.<TAB>
        layer_1.bias                layer_1.neurons
        layer_1.bias_initializer    layer_1.n_params(
        layer_1.input_shape         layer_1.output_shape(
        layer_1.kernel_initializer  layer_1.trainable
        layer_1.layer_input         layer_1.weight
        layer_1.layer_name(

        >>> layer_2 = net.layers[1]
        >>> layer_2.<TAB>
        layer_2.activation_name  layer_2.output_shape(
        layer_2.input_shape      layer_2.n_params(
        layer_2.layer_name(      layer_2.trainable

        >>> layer_3 = net.layers[-1]
        >>> layer_3.<TAB>
        layer_3.drop_rate      layer_3.n_params(
        layer_3.input_shape    layer_3.output_shape(
        layer_3.layer_name(    layer_3.trainable
        """

        return self._layers

    @layers.setter
    def layers(self, value):

        self._layers = value

    @property
    def history(self):
        """
        A dictionary saving the training and validation losses.

        The keys are

        - 'loss_train'
        - 'loss_valid'
        """

        return self._history

    @history.setter
    def history(self, value):

        self._history = value

    def add(self, layer: str, options: dict):
        """
        Add layers to the neural network.

        Parameters
        ----------
        layer : str
            Layer name. Should be one of

            - 'dense'
            - 'activation'
            - 'scale'
            - 'dropout'

        options : dict
            A dictionary to configure layers added to the network.

        Examples
        --------
        Define some concerned hydrological parameters

        >>> nd = 6  # number of hydrological descriptors
        >>> ncv = 4  # number of control vectors ("cp", "cft", "exc", "lr")
        >>> bounds = [[1.e-06,1000], [1.e-06,1000], [-50, 50], [1.e-06,1000]]  # bound constraints

        Initialize the neural network

        >>> net = smash.Net()

        Define graph

        >>> # First Dense Layer
        >>> net.add(layer="dense", options={"input_shape": (nd,), "neurons": 32})  # input_shape is only required for the first layer
        >>> # Activation funcion following the first dense layer
        >>> net.add(layer="activation", options={"name": "relu"})
        >>> # Second Dense Layer
        >>> net.add(layer="dense", options={"neurons": 16})
        >>> # Activation function following the second dense layer
        >>> net.add(layer="activation", options={"name": "relu"})
        >>> # Third Dense Layer
        >>> net.add(layer="dense", options={"neurons": ncv})
        >>> # Activation function following the third dense layer
        >>> net.add(layer="activation", options={"name": "sigmoid"})
        >>> # Scaling layer for the output of the network
        >>> net.add(layer="scale", options={"bounds": bounds})

        Compile and display a summary of the network

        >>> net.compile()
        >>> net
        +-------------+
        | Net summary |
        +-------------+
        Input Shape: (6,)
        +----------------------+--------------+---------+
        | Layer (type)         | Output Shape | Param # |
        +----------------------+--------------+---------+
        | Dense                | (32,)        | 224     |
        | Activation (ReLU)    | (32,)        | 0       |
        | Dense                | (16,)        | 528     |
        | Activation (ReLU)    | (16,)        | 0       |
        | Dense                | (4,)         | 68      |
        | Activation (Sigmoid) | (4,)         | 0       |
        | Scale (MinMaxScale)  | (4,)         | 0       |
        +----------------------+--------------+---------+
        Total params: 820
        Trainable params: 820
        Non-trainable params: 0
        """

        layer = LAYERS[layer.lower()](**options)

        if not self.layers:  # Check options if first layer

            if "input_shape" in options:

                if not isinstance(options["input_shape"], tuple):

                    raise ValueError(
                        f"input_shape option should be a tuple, not {type(options['input_shape'])}"
                    )

            else:
                raise TypeError(
                    f"First layer missing required option argument: 'input_shape'"
                )

        else:  # If be not the first layer then set the input shape to the output shape of the next added layer

            layer._set_input_shape(shape=self.layers[-1].output_shape())

        # Add layer to the network
        self.layers.append(layer)

    def compile(
        self,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        options: dict | None = None,
        random_state: int | None = None,
    ):
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

        options : dict or None, default None
            A dictionary of optimizer options.

        random_state : int or None, default None
            Random seed used to initialize weights.

            .. note::
                If not given, the weights will be initialized with a random seed.

        Examples
        --------
        >>> net = smash.Net()

        Define graph

        >>> net.add(layer="dense", options={"input_shape": (6,), "neurons": 16})
        >>> net.add(layer="activation", options={"name": "relu"})

        Compile the network

        >>> net.compile(optimizer='sgd', learning_rate=0.01, options={'momentum': 0.001})
        >>> net
        +-------------+
        | Net summary |
        +-------------+
        Input Shape: (6,)
        +-------------------+--------------+---------+
        | Layer (type)      | Output Shape | Param # |
        +-------------------+--------------+---------+
        | Dense             | (16,)        | 112     |
        | Activation (ReLU) | (16,)        | 0       |
        +-------------------+--------------+---------+
        Total params: 112
        Trainable params: 112
        Non-trainable params: 0
        """

        if options is None:
            options = {}

        if random_state is not None:
            np.random.seed(random_state)

        if self.layers:

            opt = OPTIMIZERS[optimizer.lower()](learning_rate, **options)

            for layer in self.layers:

                if hasattr(layer, "_initialize"):

                    layer._initialize(opt)

            self._compiled = True
            self._optimizer = optimizer
            self._learning_rate = learning_rate

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
                Dropout, Activation and Scaling functions do not have any weights and biases,
                so it is not important to set trainable weights at these layers.
        """

        if len(trainable) == len(self.layers):

            for i, layer in enumerate(self.layers):
                layer.trainable = trainable[i]

        else:
            raise ValueError(
                f"Inconsistent length between trainable ({len(trainable)}) and the number of layers ({len(self.layers)})"
            )

    def _fit_d2p(
        self,
        x_train: np.ndarray,
        instance: Model,
        control_vector: np.ndarray,
        mask: np.ndarray,
        parameters_bgd: ParametersDT,
        states_bgd: StatesDT,
        epochs: int,
        early_stopping: bool,
        verbose: bool,
    ):  # fit physiographic descriptors to Model parameters mapping

        if not self._compiled:
            raise ValueError(f"The network has not been compiled yet")

        loss_opt = 0  # only use for early stopping purpose

        # train model
        for epo in tqdm(range(epochs), desc="Training"):

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

                        if hasattr(layer, "_initialize"):

                            layer._weight = np.copy(layer.weight)
                            layer._bias = np.copy(layer.bias)

            # Backpropagation
            self._backward_pass(loss_grad=loss_grad)

            if verbose:
                ret = []

                ret.append(f"{' ' * 4}At epoch")
                ret.append("{:3}".format(epo + 1))
                ret.append("J =" + "{:10.6f}".format(loss))
                ret.append("|proj g| =" + "{:10.6f}".format(proj_g))

                tqdm.write((" " * 4).join(ret))

            self.history["loss_train"].append(loss)

        if early_stopping:

            for layer in self.layers:

                if hasattr(layer, "_initialize"):

                    layer.weight = np.copy(layer._weight)
                    layer.bias = np.copy(layer._bias)

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
    def _set_input_shape(self, shape: tuple):
        """Sets the shape that the layer expects of the input in the forward
        pass method"""
        self.input_shape = shape

    def layer_name(self):
        """The name of the layer. Used in model summary."""
        return self.__class__.__name__

    def n_params(self):
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

    def __init__(self, name: str, **unknown_options):

        _check_unknown_options("Activation Layer", unknown_options)

        self.input_shape = None

        self.activation_name = name
        self._activation_func = ACTIVATION_FUNC[name.lower()]()
        self.trainable = True

    def layer_name(self):
        return "Activation (%s)" % (self._activation_func.__class__.__name__)

    def _forward_pass(self, x: np.ndarray, training: bool = True):
        self.layer_input = x
        return self._activation_func(x)

    def _backward_pass(self, accum_grad: np.ndarray):
        return accum_grad * self._activation_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape


class Scale(Layer):
    """Scale function for outputs from the last layer w.r.t. parameters bounds."""

    def __init__(self, bounds: list | tuple | np.ndarray, **unknown_options):

        _check_unknown_options("Scale Layer", unknown_options)

        self.input_shape = None

        self.scale_name = "minmaxscale"

        self._scale_func = MinMaxScale(np.array(bounds))

        self.trainable = True

    def layer_name(self):
        return "Scale (%s)" % (self._scale_func.__class__.__name__)

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
    """A fully-connected NN layer.
    Parameters:
    -----------
    neurons: int
        The number of neurons in the layer.

    input_shape: tuple, default None
        The expected input shape of the layer. For dense layers a single digit specifying
        the number of features of the input. Must be specified if it is the first layer in
        the network.

    kernel_initializer : str, default glorot_uniform
        Weight initialization method. Should be one of

        - 'uniform'
        - 'glorot_uniform'
        - 'he_uniform'
        - 'normal'
        - 'glorot_normal'
        - 'he_normal'
        - 'zeros'

    bias_initializer : str, default zeros
        Bias initialization method. Should be one of

        - 'uniform'
        - 'glorot_uniform'
        - 'he_uniform'
        - 'normal'
        - 'glorot_normal'
        - 'he_normal'
        - 'zeros'
    """

    def __init__(
        self,
        neurons: int,
        input_shape: tuple | None = None,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        **unknown_options,
    ):

        _check_unknown_options("Dense Layer", unknown_options)

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
            raise ValueError(
                f"Unknown bias initializer: {self.bias_initializer}. Choices {WB_INITIALIZER}"
            )

    def _initialize(self, optimizer: function):

        # Initialize weights and biases
        _wb_initialization(self, "weight")
        _wb_initialization(self, "bias")

        # Set optimizer
        self._weight_opt = copy.copy(optimizer)
        self._bias_opt = copy.copy(optimizer)

    def n_params(self):
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
        self.weight = self._weight_opt.update(self.weight, grad_w)
        self.bias = self._bias_opt.update(self.bias, grad_w0)

        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass
        accum_grad = accum_grad.dot(weight.T)
        return accum_grad

    def output_shape(self):
        return (self.neurons,)


class Dropout(Layer):
    """Randomly sets the output of the previous layer
    to be zero with a drop rate.

    Parameters:
    -----------
    drop_rate: float
        Drop rate.
    """

    def __init__(self, drop_rate: float, **unknown_options):

        _check_unknown_options("Dropout Layer", unknown_options)

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


LAYERS = {
    "dense": Dense,
    "activation": Activation,
    "scale": Scale,
    "dropout": Dropout,
}


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
    def __init__(self, bounds: np.ndarray):

        self._bounds = bounds

        self.lower = np.array([b[0] for b in bounds])
        self.upper = np.array([b[1] for b in bounds])

    def __call__(self, x: np.ndarray):
        return self.lower + x * (self.upper - self.lower)

    def gradient(self, x: np.ndarray):
        return self.upper - self.lower


### OPTIMIZER ###


class StochasticGradientDescent:
    def __init__(self, learning_rate: float, momentum: float = 0, **unknown_options):

        _check_unknown_options("SGD optimizer", unknown_options)

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
    def __init__(
        self,
        learning_rate: float,
        b1: float = 0.9,
        b2: float = 0.999,
        **unknown_options,
    ):

        _check_unknown_options("Adam optimizer", unknown_options)

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
    def __init__(self, learning_rate, **unknown_options):

        _check_unknown_options("Adagrad optimizer", unknown_options)

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
    def __init__(self, learning_rate: float, rho: float = 0.9, **unknown_options):

        _check_unknown_options("RMSprop optimizer", unknown_options)

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

    #% Set parameters or states
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
            getattr(parameters_b, name)[mask]
            if name in instance.setup._parameters_name
            else getattr(states_b, name)[mask]
            for name in control_vector
        ]
    )

    return grad


### OTHERS ###


def _inf_norm(grad: np.ndarray):

    return np.amax(np.abs(grad))


def _check_unknown_options(type_check: str, unknown_options: dict):

    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        warnings.warn("Unknown %s options: '%s'" % (type_check, msg))
