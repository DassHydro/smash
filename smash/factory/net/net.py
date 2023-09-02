from __future__ import annotations

from smash._typing import Numeric

from smash._constant import PY_OPTIMIZER, PY_OPTIMIZER_CLASS

from smash.factory.net._standardize import _standardize_add_args
from smash.factory.net._loss import _hcost, _hcost_prime, _inf_norm
from smash.factory.net._layers import Dense, Activation, Scale, Dropout
from smash.factory.net._optimizers import SGD, Adam, Adagrad, RMSprop

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.fcore._mwd_options import OptionsDT
    from smash.fcore._mwd_returns import ReturnsDT

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
    >>> from smash.factory import Net
    >>> net = Net()
    >>> net
    The network does not contain layers or has not been compiled yet
    """

    def __init__(self):
        self.layers = []

        self.history = {"loss_train": [], "loss_valid": []}

        self._opt = None

    def __repr__(self):
        ret = []

        tab = [["Layer Type", "Input/Output Shape", "Num Parameters"]]

        tot_params = 0
        trainable_params = 0

        for layer in self.layers:
            layer_name = layer.layer_name()

            n_params = layer.n_params()

            ioshape = f"{layer.input_shape}/{layer.output_shape()}"

            tab.append([layer_name, str(ioshape), str(n_params)])

            tot_params += n_params

            if layer.trainable:
                trainable_params += n_params

        table_instance = AsciiTable(tab)
        table_instance.inner_column_border = False
        table_instance.padding_left = 1
        table_instance.padding_right = 1

        ret.append(table_instance.table)
        ret.append(f"Total parameters: {tot_params}")
        ret.append(f"Trainable parameters: {trainable_params}")

        return "\n".join(ret)

    @property
    def layers(self):
        """
        List of Layer objects defining the graph of the network.

        The graph is set using `smash.factory.Net.add` method.

        Examples
        --------
        >>> from smash.factory import Net
        >>> net = Net()
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

            .. hint::
                See options for each layer type:

                - 'dense' :ref:`(see here) <api_reference.add_dense>`
                - 'activation' :ref:`(see here) <api_reference.add_activation>`
                - 'scale' :ref:`(see here) <api_reference.add_scale>`
                - 'dropout' :ref:`(see here) <api_reference.add_dropout>`

        Examples
        --------
        Initialize the neural network

        >>> from smash.factory import Net
        >>> net = Net()

        Define graph

        >>> # First Dense Layer
        >>> # input_shape is only required for the first layer
        >>> net.add(layer="dense", options={"input_shape": (8,), "neurons": 32})
        >>> # Activation funcion following the first dense layer
        >>> net.add(layer="activation", options={"name": "relu"})
        >>> # Second Dense Layer
        >>> net.add(layer="dense", options={"neurons": 16})
        >>> # Activation function following the second dense layer
        >>> net.add(layer="activation", options={"name": "relu"})
        >>> # Third Dense Layer
        >>> net.add(layer="dense", options={"neurons": 4})
        >>> # Last Activation function (output of the network)
        >>> net.add(layer="activation", options={"name": "sigmoid"})

        Compile and display a summary of the network

        >>> net.compile()
        >>> net
        +----------------------------------------------------------+
        | Layer Type            Input/Output Shape  Num Parameters |
        +----------------------------------------------------------+
        | Dense                 (8,)/(32,)          288            |
        | Activation (ReLU)     (32,)/(32,)         0              |
        | Dense                 (32,)/(16,)         528            |
        | Activation (ReLU)     (16,)/(16,)         0              |
        | Dense                 (16,)/(4,)          68             |
        | Activation (Sigmoid)  (4,)/(4,)           0              |
        +----------------------------------------------------------+
        Total parameters: 884
        Trainable parameters: 884
        Optimizer: (adam, lr=0.001)
        """

        layer, options = _standardize_add_args(layer, options)

        lay = eval(layer)(**options)

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
            lay._set_input_shape(shape=self.layers[-1].output_shape())

        # Add layer to the network
        self.layers.append(lay)

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
                Dropout, activation, and scaling functions are non-parametric layers,
                meaning they do not have any learnable weights or biases.
                Therefore, it is not necessary to set these layers as trainable
                since they do not involve any weight updates during training.
        """

        if len(trainable) == len(self.layers):
            for i, layer in enumerate(self.layers):
                layer.trainable = trainable[i]

        else:
            raise ValueError(
                f"Inconsistent size between trainable ({len(trainable)}) and the number of layers ({len(self.layers)})"
            )

    def _compile(
        self,
        optimizer: str,
        learning_param: dict,
        random_state: Numeric | None,
    ):
        """
        Compile the network and set optimizer.

        Parameters
        ----------
        optimizer : str, default 'adam'
            Name of optimizer. Should be one of

            - 'sgd'
            - 'adam'
            - 'adagrad'
            - 'rmsprop'

        options : dict or None, default None
            A dictionary of optimizer options.

            .. hint::
                See options for each optimizer:

                - 'sgd' :ref:`(see here) <api_reference.compile_sgd>`
                - 'adam' :ref:`(see here) <api_reference.compile_adam>`
                - 'adagrad' :ref:`(see here) <api_reference.compile_adagrad>`
                - 'rmsprop' :ref:`(see here) <api_reference.compile_rmsprop>`

        random_state : int or None, default None
            Random seed used to initialize weights.

            .. note::
                If not given, the weights will be initialized with a random seed.

        Examples
        --------
        >>> from smash.factory import Net
        >>> net = Net()

        Define graph

        >>> net.add(layer="dense", options={"input_shape": (6,), "neurons": 16})
        >>> net.add(layer="activation", options={"name": "relu"})

        Compile the network

        >>> net.compile(optimizer='sgd', options={'learning_rate': 0.009, 'momentum': 0.001})
        >>> net
        +-------------------------------------------------------+
        | Layer Type         Input/Output Shape  Num Parameters |
        +-------------------------------------------------------+
        | Dense              (6,)/(16,)          112            |
        | Activation (ReLU)  (16,)/(16,)         0              |
        +-------------------------------------------------------+
        Total parameters: 112
        Trainable parameters: 112
        Optimizer: (sgd, lr=0.009)
        """

        if self.layers:
            ind = PY_OPTIMIZER.index(optimizer.lower())

            func = eval(PY_OPTIMIZER_CLASS[ind])

            opt = func(**learning_param)

            if random_state is not None:
                np.random.seed(random_state)

            for layer in self.layers:
                if hasattr(layer, "_initialize"):
                    layer._initialize(opt)

            self._opt = opt

        else:
            raise ValueError("The network does not contain layers")

    def _fit_d2p(
        self,
        x_train: np.ndarray,
        mask: np.ndarray,
        instance: Model,
        wrap_options: OptionsDT,
        wrap_returns: ReturnsDT,
        optimizer: str,
        parameters: np.ndarray,
        learning_rate: Numeric,
        random_state: Numeric | None,
        epochs: int,
        early_stopping: bool,
        verbose: bool,
    ):  # fit physiographic descriptors to Model parameters mapping
        # % compile net
        self._compile(
            optimizer=optimizer,
            learning_param={"learning_rate": learning_rate},
            random_state=random_state,
        )

        loss_opt = 0  # only use for early stopping purpose

        # % train model
        for epo in tqdm(range(epochs), desc="    Training"):
            # forward propogation
            y_pred = self._forward_pass(x_train)

            # calculate the gradient of the loss function wrt y_pred
            loss_grad = _hcost_prime(
                y_pred, parameters, mask, instance, wrap_options, wrap_returns
            )

            # compute loss
            loss = _hcost(instance)

            # calculate the infinity norm of the projected gradient
            proj_g = _inf_norm(loss_grad)

            # early stopping
            if early_stopping:
                if loss_opt > loss or epo == 0:
                    loss_opt = loss

                    for layer in self.layers:
                        if hasattr(layer, "_initialize"):
                            layer._weight = np.copy(layer.weight)
                            layer._bias = np.copy(layer.bias)

            # backpropagation
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
