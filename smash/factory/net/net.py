from __future__ import annotations

from typing import TYPE_CHECKING

from smash._constant import OPTIMIZABLE_NN_PARAMETERS, PY_OPTIMIZER, PY_OPTIMIZER_CLASS
from smash.factory.net._layers import Activation, Conv2D, Dense, Dropout, Flatten, Scale
from smash.factory.net._loss import _hcost, _hcost_prime, _inf_norm

# Used inside eval statement
from smash.factory.net._optimizers import SGD, Adagrad, Adam, RMSprop  # noqa: F401
from smash.factory.net._standardize import (
    _standardize_add_conv2d_args,
    _standardize_add_dense_args,
    _standardize_add_dropout_args,
    _standardize_add_scale_args,
    _standardize_set_trainable_args,
)

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.fcore._mwd_options import OptionsDT
    from smash.fcore._mwd_returns import ReturnsDT
    from smash.util._typing import Numeric

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
    +------------------------------------------------+
    | Layer Type  Input/Output Shape  Num Parameters |
    +------------------------------------------------+
    Total parameters: 0
    Trainable parameters: 0
    """

    def __init__(self):
        self.layers = []

        self.history = {"loss_train": [], "loss_valid": [], "proj_grad": []}

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

        Examples
        --------
        >>> from smash.factory import Net
        >>> net = Net()
        >>> net.add_dense(32, input_shape=6, activation="relu")
        >>> net.add_dropout(0.2)
        >>> net
        +-------------------------------------------------------+
        | Layer Type         Input/Output Shape  Num Parameters |
        +-------------------------------------------------------+
        | Dense              (6,)/(32,)          224            |
        | Activation (ReLU)  (32,)/(32,)         0              |
        | Dropout            (32,)/(32,)         0              |
        +-------------------------------------------------------+
        Total parameters: 224
        Trainable parameters: 224

        If you are using IPython, tab completion allows you to visualize all the attributes and methods of
        each Layer object:

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

        >>> layer_3 = net.layers[2]
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
        A dictionary saving training information.

        The keys are 'loss_train', 'loss_valid', and 'proj_grad'.
        """

        return self._history

    @history.setter
    def history(self, value):
        self._history = value

    def add_dense(
        self,
        neurons: int,
        input_shape: int | tuple | list | None = None,
        activation: str | None = None,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
    ):
        """
        Add a fully-connected layer to the neural network.

        Parameters
        ----------
        neurons : int
            The number of neurons in the layer.

        input_shape : int, tuple, list, or None, default None
            The expected input shape of the layer.
            It must be specified if this is the first layer in the network.

        activation : str or None, default None
            Add an activation layer following the current layer if specified. Should be one of

            - ``'relu'`` : Rectified Linear Unit
            - ``'sigmoid'`` : Sigmoid
            - ``'selu'`` : Scaled Exponential Linear Unit
            - ``'elu'`` : Exponential Linear Unit
            - ``'softmax'`` : Softmax
            - ``'leakyrelu'`` : Leaky Rectified Linear Unit
            - ``'tanh'`` : Hyperbolic Tangent
            - ``'softplus'`` : Softplus

        kernel_initializer : str, default 'glorot_uniform'
            Kernel initialization method. Should be one of ``'uniform'``, ``'glorot_uniform'``,
            ``'he_uniform'``, ``'normal'``, ``'glorot_normal'``, ``'he_normal'``, ``'zeros'``.

        bias_initializer : str, default 'zeros'
            Bias initialization method. Should be one of ``'uniform'``, ``'glorot_uniform'``,
            ``'he_uniform'``, ``'normal'``, ``'glorot_normal'``, ``'he_normal'``, ``'zeros'``.

        Examples
        --------
        >>> from smash.factory import Net
        >>> net = Net()
        >>> net.add_dense(128, input_shape=12, activation="relu")
        >>> net.add_dense(32, activation="sigmoid")
        >>> net
        +----------------------------------------------------------+
        | Layer Type            Input/Output Shape  Num Parameters |
        +----------------------------------------------------------+
        | Dense                 (12,)/(128,)        1664           |
        | Activation (ReLU)     (128,)/(128,)       0              |
        | Dense                 (128,)/(32,)        4128           |
        | Activation (Sigmoid)  (32,)/(32,)         0              |
        +----------------------------------------------------------+
        Total parameters: 5792
        Trainable parameters: 5792
        """

        neurons, input_shape, activation, kernel_initializer, bias_initializer = _standardize_add_dense_args(
            self, neurons, input_shape, activation, kernel_initializer, bias_initializer
        )

        # Add Dense layer to the network
        self.layers.append(Dense(neurons, input_shape, kernel_initializer, bias_initializer))

        # Add activation layer if specified
        if activation is not None:
            layer = Activation(activation)
            layer._set_input_shape(shape=self.layers[-1].output_shape())

            self.layers.append(layer)

    def add_conv2d(
        self,
        filters: int,
        filter_shape: int | tuple,
        input_shape: tuple | list | None = None,
        activation: str | None = None,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
    ):
        """
        Add a 2D convolutional layer (with same padding and a stride of one) to the neural network.

        Parameters
        ----------
        filters : int
            The number of filters in the convolutional layer.

        filter_shape : int or tuple
            The size of the convolutional window.

        input_shape : tuple, list, or None, default None
            The expected input shape of the layer.
            It must be specified if this is the first layer in the network.

        activation : str or None, default None
            Add an activation layer following the current layer if specified. Should be one of

            - ``'relu'`` : Rectified Linear Unit
            - ``'sigmoid'`` : Sigmoid
            - ``'selu'`` : Scaled Exponential Linear Unit
            - ``'elu'`` : Exponential Linear Unit
            - ``'softmax'`` : Softmax
            - ``'leakyrelu'`` : Leaky Rectified Linear Unit
            - ``'tanh'`` : Hyperbolic Tangent
            - ``'softplus'`` : Softplus

        kernel_initializer : str, default 'glorot_uniform'
            Kernel initialization method. Should be one of ``'uniform'``, ``'glorot_uniform'``,
            ``'he_uniform'``, ``'normal'``, ``'glorot_normal'``, ``'he_normal'``, ``'zeros'``.

        bias_initializer : str, default 'zeros'
            Bias initialization method. Should be one of ``'uniform'``, ``'glorot_uniform'``,
            ``'he_uniform'``, ``'normal'``, ``'glorot_normal'``, ``'he_normal'``, ``'zeros'``.

        Examples
        --------
        >>> from smash.factory import Net
        >>> net = Net()
        >>> net.add_conv2d(128, filter_shape=(8, 6), input_shape=(56, 50, 3), activation="relu")
        >>> net.add_conv2d(32, filter_shape=(8, 6), activation="leakyrelu")
        >>> net
        +---------------------------------------------------------------------+
        | Layer Type              Input/Output Shape           Num Parameters |
        +---------------------------------------------------------------------+
        | Conv2D                  (56, 50, 3)/(56, 50, 128)    18560          |
        | Activation (ReLU)       (56, 50, 128)/(56, 50, 128)  0              |
        | Conv2D                  (56, 50, 128)/(56, 50, 32)   196640         |
        | Activation (LeakyReLU)  (56, 50, 32)/(56, 50, 32)    0              |
        +---------------------------------------------------------------------+
        Total parameters: 215200
        Trainable parameters: 215200
        """

        (
            filters,
            filter_shape,
            input_shape,
            activation,
            kernel_initializer,
            bias_initializer,
        ) = _standardize_add_conv2d_args(
            self, filters, filter_shape, input_shape, activation, kernel_initializer, bias_initializer
        )

        # Add Conv2D layer to the network
        self.layers.append(Conv2D(filters, filter_shape, input_shape, kernel_initializer, bias_initializer))

        # Add activation layer if specified
        if activation is not None:
            layer = Activation(activation)
            layer._set_input_shape(shape=self.layers[-1].output_shape())

            self.layers.append(layer)

    def add_scale(self, bounds: list):
        """
        Add a scaling layer that applies the min-max scaling function to the outputs.

        Parameters
        ----------
        bounds : ListLike
            A sequence of ``(min, max)`` values that the outputs will be scaled to.

        Examples
        --------
        >>> from smash.factory import Net
        >>> net = Net()
        >>> net.add_dense(2, input_shape=8, activation="sigmoid")
        >>> net.add_scale([(2,3), (5,6)])
        >>> net
        +----------------------------------------------------------+
        | Layer Type            Input/Output Shape  Num Parameters |
        +----------------------------------------------------------+
        | Dense                 (8,)/(2,)           18             |
        | Activation (Sigmoid)  (2,)/(2,)           0              |
        | Scale (MinMaxScale)   (2,)/(2,)           0              |
        +----------------------------------------------------------+
        Total parameters: 18
        Trainable parameters: 18
        """

        bounds, bound_in = _standardize_add_scale_args(self, bounds)

        layer = Scale(bounds, bound_in)
        layer._set_input_shape(shape=self.layers[-1].output_shape())

        self.layers.append(layer)

    def add_flatten(self):
        """
        Add a flatten layer to reshape the input from 2D layer into 1D layer.

        Examples
        --------
        >>> from smash.factory import Net
        >>> net = Net()
        >>> net.add_conv2d(64, filter_shape=5, input_shape=(30, 32, 6))
        >>> net.add_flatten()
        >>> net.add_dense(16)
        +------------------------------------------------------+
        | Layer Type  Input/Output Shape        Num Parameters |
        +------------------------------------------------------+
        | Conv2D      (30, 32, 6)/(30, 32, 64)  9664           |
        | Flatten     (30, 32, 64)/(960, 64)    0              |
        | Dense       (960, 64)/(960, 16)       1040           |
        +------------------------------------------------------+
        Total parameters: 10704
        Trainable parameters: 10704
        """

        layer = Flatten()
        layer._set_input_shape(shape=self.layers[-1].output_shape())

        self.layers.append(layer)

    def add_dropout(self, drop_rate: float):
        """
        Add a dropout layer that randomly sets the output of the previous layer to zero
        with a specified probability.

        Parameters
        ----------
        drop_rate : float
            The probability of setting a given output value to zero.

        Examples
        --------
        >>> from smash.factory import Net
        >>> net = Net()
        >>> net.add_dense(1024, input_shape=56)
        >>> net.add_dropout(0.25)
        >>> net
        +------------------------------------------------+
        | Layer Type  Input/Output Shape  Num Parameters |
        +------------------------------------------------+
        | Dense       (56,)/(1024,)       58368          |
        | Dropout     (1024,)/(1024,)     0              |
        +------------------------------------------------+
        Total parameters: 58368
        Trainable parameters: 58368
        """

        drop_rate = _standardize_add_dropout_args(drop_rate)

        layer = Dropout(drop_rate)
        layer._set_input_shape(shape=self.layers[-1].output_shape())

        self.layers.append(layer)

    def copy(self):
        """
        Make a deepcopy of the Net.

        Returns
        -------
        net : Net
            A copy of Net.
        """

        return copy.deepcopy(self)

    def set_trainable(self, trainable: list[bool]):
        """
        Method which enables to train or freeze the weights and biases of the network's layers.

        Parameters
        ----------
        trainable : ListLike
            List of booleans with a length of the total number of the network's layers.

        Examples
        --------
        >>> from smash.factory import Net
        >>> net = Net()
        >>> net.add_dense(32, input_shape=8, activation="relu")
        >>> net.add_dense(16, activation="relu")
        >>> net
        +-------------------------------------------------------+
        | Layer Type         Input/Output Shape  Num Parameters |
        +-------------------------------------------------------+
        | Dense              (8,)/(32,)          288            |
        | Activation (ReLU)  (32,)/(32,)         0              |
        | Dense              (32,)/(16,)         528            |
        | Activation (ReLU)  (16,)/(16,)         0              |
        +-------------------------------------------------------+
        Total parameters: 816
        Trainable parameters: 816

        Freeze the parameters in the second dense layer:

        >>> net.set_trainable([1, 0, 0, 0])
        >>> net
        +-------------------------------------------------------+
        | Layer Type         Input/Output Shape  Num Parameters |
        +-------------------------------------------------------+
        | Dense              (8,)/(32,)          288            |
        | Activation (ReLU)  (32,)/(32,)         0              |
        | Dense              (32,)/(16,)         528            |
        | Activation (ReLU)  (16,)/(16,)         0              |
        +-------------------------------------------------------+
        Total parameters: 816
        Trainable parameters: 288
        """

        trainable = _standardize_set_trainable_args(self, trainable)

        for i, layer in enumerate(self.layers):
            layer.trainable = trainable[i]

    def _compile(
        self,
        optimizer: str,
        learning_param: dict,
        random_state: Numeric | None,
    ):
        """
        Private function: Compile the neural network.
        """
        ind = PY_OPTIMIZER.index(optimizer.lower())

        func = eval(PY_OPTIMIZER_CLASS[ind])

        opt = func(**learning_param)

        if random_state is not None:
            np.random.seed(random_state)

        for layer in self.layers:
            if hasattr(layer, "_initialize"):
                layer._initialize(opt)

        self._opt = opt

    def _fit_d2p(
        self,
        x_train: np.ndarray,
        instance: Model,
        wrap_options: OptionsDT,
        wrap_returns: ReturnsDT,
        optimizer: str,
        parameters: np.ndarray,
        learning_rate: Numeric,
        random_state: Numeric | None,
        epochs: int,
        early_stopping: int,
        verbose: bool,
    ):
        """
        Private function: fit physiographic descriptors to Model parameters mapping.
        """
        # % Compile net
        self._compile(
            optimizer=optimizer,
            learning_param={"learning_rate": learning_rate},
            random_state=random_state,
        )

        # % Initialize optimizer for the parameterization NN if used
        ind = PY_OPTIMIZER.index(optimizer.lower())
        func = eval(PY_OPTIMIZER_CLASS[ind])

        opt_nn_parameters = [func(learning_rate=learning_rate) for _ in range(2 * instance.setup.n_layers)]

        # % Train model
        for epo in tqdm(range(epochs), desc="    Training"):
            # forward propogation
            y_pred = self._forward_pass(x_train)

            # calculate the gradient of the loss function wrt y_pred of the regionalization NN
            # and get the gradient of the parameterization NN if used
            init_loss_grad, nn_parameters_b = _hcost_prime(
                y_pred, parameters, instance, wrap_options, wrap_returns
            )

            # compute loss
            loss = _hcost(instance)
            self.history["loss_train"].append(loss)

            # save optimal weights if early stopping is used
            if early_stopping:
                if epo == 0:
                    loss_opt = {"epo": 0, "value": loss}
                    nn_parameters_bak = instance.nn_parameters.copy()

                if loss <= loss_opt["value"]:
                    loss_opt["epo"] = epo
                    loss_opt["value"] = loss

                    # backup nn_parameters
                    nn_parameters_bak = instance.nn_parameters.copy()

                    # backup weights and biases of rr_parameters
                    for layer in self.layers:
                        if hasattr(layer, "_initialize"):
                            layer._weight = np.copy(layer.weight)
                            layer._bias = np.copy(layer.bias)

                else:
                    if (
                        epo - loss_opt["epo"] > early_stopping
                    ):  # stop training if the loss values do not decrease through early_stopping consecutive
                        # epochs
                        break

            # backpropagation and weights update
            if epo < epochs - 1:
                for i, key in enumerate(OPTIMIZABLE_NN_PARAMETERS[max(0, instance.setup.n_layers - 1)]):
                    if key in parameters:  # update trainable parameters of the parameterization NN if used
                        setattr(
                            instance.nn_parameters,
                            key,
                            opt_nn_parameters[i].update(
                                getattr(instance.nn_parameters, key), nn_parameters_b[i]
                            ),
                        )

                # backpropagation and update weights of the regionalization NN
                loss_grad = self._backward_pass(init_loss_grad, inplace=True)
            else:  # do not update weights at the last epoch
                loss_grad = self._backward_pass(init_loss_grad, inplace=False)

            # calculate projected gradient for the LPR operator
            self.history["proj_grad"].append(_inf_norm([loss_grad, nn_parameters_b]))

            if verbose:
                ret = []

                ret.append(f"{' ' * 4}At epoch")
                ret.append("{:3}".format(epo + 1))
                ret.append("J =" + "{:10.6f}".format(loss))
                ret.append("|proj g| =" + "{:10.6f}".format(self.history["proj_grad"][-1]))

                tqdm.write((" " * 4).join(ret))

        if early_stopping:
            instance.nn_parameters = nn_parameters_bak  # revert nn_parameters

            for layer in self.layers:
                if hasattr(layer, "_initialize"):  # revert weights and biases of rr_parameters
                    layer.weight = np.copy(layer._weight)
                    layer.bias = np.copy(layer._bias)

                    # remove tmp attr for each layer of net
                    del layer._weight
                    del layer._bias

    def _forward_pass(self, x_train: np.ndarray):
        layer_output = x_train

        for layer in self.layers:
            layer_output = layer._forward_pass(layer_output)

        return layer_output

    def _backward_pass(self, loss_grad: np.ndarray, inplace=True):
        if inplace:
            net = self
        else:
            net = self.copy()

        for layer in reversed(net.layers):
            loss_grad = layer._backward_pass(loss_grad)

        return loss_grad
