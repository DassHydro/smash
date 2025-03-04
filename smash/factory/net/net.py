from __future__ import annotations

from typing import TYPE_CHECKING

from smash._constant import ADAPTIVE_OPTIMIZER, OPTIMIZABLE_NN_PARAMETERS, OPTIMIZER_CLASS
from smash.core.simulation.optimize._tools import _inf_norm, _net2vect
from smash.core.simulation.optimize.optimize import Optimize
from smash.factory.net._layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Scale,
    _set_initialized_wb_to_layer,
)
from smash.factory.net._loss import _get_cost_value, _get_gradient_value

# Used inside eval statement
from smash.factory.net._optimizers import SGD, Adagrad, Adam, RMSprop  # noqa: F401
from smash.factory.net._standardize import (
    _standardize_add_conv2d_args,
    _standardize_add_dense_args,
    _standardize_add_dropout_args,
    _standardize_add_scale_args,
    _standardize_forward_pass_args,
    _standardize_set_bias_args,
    _standardize_set_trainable_args,
    _standardize_set_weight_args,
)
from smash.fcore._mwd_parameters_manipulation import (
    parameters_to_control as wrap_parameters_to_control,
)

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.fcore._mwd_options import OptionsDT
    from smash.fcore._mwd_parameters import ParametersDT
    from smash.fcore._mwd_returns import ReturnsDT
    from smash.util._typing import Any, Numeric

import copy

import numpy as np
from terminaltables import AsciiTable

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

        self.history = {"loss_train": [], "proj_grad": []}

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
        layer_1.bias                layer_1.n_params()
        layer_1.bias_initializer    layer_1.neurons
        layer_1.bias_shape          layer_1.output_shape()
        layer_1.input_shape         layer_1.trainable
        layer_1.kernel_initializer  layer_1.weight
        layer_1.layer_input         layer_1.weight_shape
        layer_1.layer_name()

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

        The keys are ``'loss_train'`` and ``'proj_grad'``.
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

        This method adds a dense layer into the neural network graph but does not initialize its weight
        and bias values.

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
            - ``'silu'`` : Sigmoid Linear Unit

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

        This method adds a 2D convolutional layer into the neural network graph but does not initialize
        its weight and bias values.

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
            - ``'silu'`` : Sigmoid Linear Unit

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
        >>> net
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
        ind = ADAPTIVE_OPTIMIZER.index(optimizer)

        func = eval(OPTIMIZER_CLASS[ind])

        opt = func(**learning_param)

        if random_state is not None:
            np.random.seed(random_state)

        for layer in self.layers:
            if hasattr(layer, "_initialize"):
                layer._initialize(opt)

        # % Reset random seed if random_state is previously set
        if random_state is not None:
            np.random.seed(None)

    def _fit_d2p(
        self,
        x_train: np.ndarray,
        instance: Model,
        parameters: ParametersDT,
        wrap_options: OptionsDT,
        wrap_returns: ReturnsDT,
        optimizer: str,
        calibrated_parameters: np.ndarray,
        learning_rate: Numeric,
        random_state: Numeric | None,
        maxiter: int,
        early_stopping: int,
        verbose: bool,
        callback: callable | None,
    ) -> int:
        """
        Private function: fit physiographic descriptors to Model parameters mapping.
        """
        # % Compile net
        self._compile(
            optimizer=optimizer,
            learning_param={"learning_rate": learning_rate},
            random_state=random_state,
        )

        # % First evaluation
        # calculate the gradient of J wrt rr_parameters and rr_initial_states
        # that are the output of the descriptors-to-parameters (d2p) NN
        # and get the gradient of the pmtz NN (pmtz) if used
        grad_d2p_init, grad_pmtz = _get_gradient_value(
            self, x_train, calibrated_parameters, instance, parameters, wrap_options, wrap_returns
        )
        grad_d2p = self._backward_pass(grad_d2p_init, inplace=False)  # do not update weight and bias

        projg = _inf_norm([grad_d2p, grad_pmtz])

        # calculate cost
        cost = _get_cost_value(instance)  # forward_run to update cost inside _get_gradient_value

        if verbose:
            print(f"{' ' * 4}At iterate {0:>5}    nfg = {1:>5}    J = {cost:>.5e}    |proj g| = {projg:>.5e}")

        # % Early stopping
        istop = 0
        opt_info = {"cost": np.inf}  # only used for early_stopping

        # % Initialize optimizer for the pmtz NN if used
        ind = ADAPTIVE_OPTIMIZER.index(optimizer)
        func = eval(OPTIMIZER_CLASS[ind])

        opt_nn_parameters = [func(learning_rate=learning_rate) for _ in range(2 * instance.setup.n_layers)]

        # % Train model
        for ite in range(1, maxiter + 1):
            # backpropagation and weights update
            for i, key in enumerate(OPTIMIZABLE_NN_PARAMETERS[max(0, instance.setup.n_layers - 1)]):
                if key in calibrated_parameters:  # update trainable parameters of the pmtz NN if used
                    setattr(
                        parameters.nn_parameters,
                        key,
                        opt_nn_parameters[i].update(getattr(parameters.nn_parameters, key), grad_pmtz[i]),
                    )

            self._backward_pass(grad_d2p_init, inplace=True)  # update weights of the d2p NN

            # cost and gradient computation
            grad_d2p_init, grad_pmtz = _get_gradient_value(
                self, x_train, calibrated_parameters, instance, parameters, wrap_options, wrap_returns
            )
            grad_d2p = self._backward_pass(grad_d2p_init, inplace=False)  # do not update weight and bias

            projg = _inf_norm([grad_d2p, grad_pmtz])

            cost = _get_cost_value(instance)  # forward_run to update cost inside _get_gradient_value

            # save optimal parameters if early stopping is used
            if early_stopping:
                if cost < opt_info["cost"] or ite == 1:
                    opt_info["ite"] = ite
                    opt_info["cost"] = cost

                    # backup nn_parameters
                    opt_info["nn_parameters"] = parameters.nn_parameters.copy()

                    # backup net
                    opt_info["net_layers"] = self.copy().layers

                elif (
                    ite - opt_info["ite"] > early_stopping
                ):  # stop training if the cost values do not decrease through early_stopping consecutive
                    # iterations
                    if verbose:
                        print(
                            f"{' ' * 4}EARLY STOPPING: NO IMPROVEMENT for {early_stopping} CONSECUTIVE "
                            f"ITERATIONS"
                        )
                    break

            self.history["proj_grad"].append(projg)
            self.history["loss_train"].append(cost)

            if callback is not None:
                wrap_parameters_to_control(
                    instance.setup,
                    instance.mesh,
                    instance._input_data,
                    parameters,
                    wrap_options,
                )
                control = np.append(parameters.control.x, _net2vect(self))

                callback(
                    iopt=Optimize(
                        {
                            "control_vector": control,
                            "cost": cost,
                            "n_iter": ite,
                            "projg": projg,
                            "net": self.copy(),
                        }
                    )
                )

            if verbose:
                print(
                    f"{' ' * 4}At iterate {ite:>5}    nfg = {ite + 1:>5}    J = {cost:>.5e}    "
                    f"|proj g| = {projg:>.5e}"
                )

                if ite == maxiter:
                    print(f"{' ' * 4}STOP: TOTAL NO. of ITERATIONS REACHED LIMIT")

        if early_stopping:
            if opt_info["ite"] < maxiter:
                if verbose:
                    print(
                        f"{' ' * 4}Revert to iteration {opt_info['ite']} with "
                        f"J = {opt_info['cost']:.5e} due to early stopping"
                    )

                parameters.nn_parameters = opt_info["nn_parameters"]  # revert nn_parameters

                self.layers = opt_info["net_layers"]  # revert net

                istop = opt_info["ite"]

        return istop

    def set_weight(self, value: list[Any] | None = None, random_state: int | None = None):
        """
        Set the values of the weight in the neural network `Net`.

        Parameters
        ----------
        value : list[`float` or `numpy.ndarray`] or None, default None
            The list of values to set to the weights of all layers. If an element of the list is
            a `numpy.ndarray`, its shape must be broadcastable into the weight shape of that layer.
            If not used, initialization methods defined in trainable layers will be used with
            a random or specified seed depending on **random_state**.

        random_state : `int` or None, default None
            Random seed used for the initialization method defined in each trainable layer.
            Only used if **value** is not set.

            .. note::
                If not given, the parameters will be initialized with a random seed.

        See Also
        --------
        Net.get_weight : Get the weights of the trainable layers of the neural network `Net`.

        Examples
        --------
        >>> from smash.factory import Net
        >>> net = Net()
        >>> net.add_dense(2, input_shape=3, kernel_initializer="uniform")
        >>> net
        +-------------------------------------------------------+
        | Layer Type         Input/Output Shape  Num Parameters |
        +-------------------------------------------------------+
        | Dense              (3,)/(2,)           8              |
        +-------------------------------------------------------+
        Total parameters: 8
        Trainable parameters: 8

        Set weights with specified values

        >>> import numpy as np
        >>> net.set_weight([np.array([[1, 2, 3], [4, 5, 6]])])

        Get the weight values

        >>> net.get_weight()
        [array([[1, 2, 3],
                [4, 5, 6]])]

        Set random weights

        >>> net.set_weight(random_state=0)
        >>> net.get_weight()
        [array([[ 0.05636498,  0.24847928,  0.11866093],
                [ 0.05182664, -0.08815584,  0.16846401]])]
        """

        value, random_state = _standardize_set_weight_args(self, value, random_state)

        if value is None:
            if random_state is not None:
                np.random.seed(random_state)

            for layer in self.layers:
                if hasattr(layer, "weight"):
                    _set_initialized_wb_to_layer(layer, "weight")

            # % Reset random seed if random_state is previously set
            if random_state is not None:
                np.random.seed(None)

        else:
            i = 0
            for layer in self.layers:
                if hasattr(layer, "weight"):
                    layer.weight = value[i]
                    i += 1

    def set_bias(self, value: list[Any] | None = None, random_state: int | None = None):
        """
        Set the values of the bias in the neural network `Net`.

        Parameters
        ----------
        value : list[`float` or `numpy.ndarray`] or None, default None
            The list of values to set to the biases of all layers. If an element of the list is
            a `numpy.ndarray`, its shape must be broadcastable into the bias shape of that layer.
            If not used, initialization methods defined in trainable layers will be used with
            a random or specified seed depending on **random_state**.

        random_state : `int` or None, default None
            Random seed used for the initialization method defined in each trainable layer.
            Only used if **value** is not set.

            .. note::
                If not given, the parameters will be initialized with a random seed.

        See Also
        --------
        Net.get_bias : Get the biases of the trainable layers of the neural network `Net`.

        Examples
        --------
        >>> from smash.factory import Net
        >>> net = Net()
        >>> net.add_dense(4, input_shape=3, activation="tanh")
        >>> net.add_dense(2, bias_initializer="he_normal")
        >>> net
        +-------------------------------------------------------+
        | Layer Type         Input/Output Shape  Num Parameters |
        +-------------------------------------------------------+
        | Dense              (3,)/(4,)           16             |
        | Activation (TanH)  (4,)/(4,)           0              |
        | Dense              (4,)/(2,)           10             |
        +-------------------------------------------------------+
        Total parameters: 26
        Trainable parameters: 26

        Set biases with specified values

        >>> net.set_bias([1.2, 1.3])

        Get the bias values

        >>> net.get_bias()
        [array([[1.2, 1.2, 1.2, 1.2]]), array([[1.3, 1.3]])]

        Set random biases

        >>> net.set_bias(random_state=0)
        >>> net.get_bias()  # default bias initializer is zeros
        [array([[0., 0., 0., 0.]]), array([[2.49474675, 0.56590775]])]
        """

        value, random_state = _standardize_set_bias_args(self, value, random_state)

        if value is None:
            if random_state is not None:
                np.random.seed(random_state)

            for layer in self.layers:
                if hasattr(layer, "bias"):
                    _set_initialized_wb_to_layer(layer, "bias")

            # % Reset random seed if random_state is previously set
            if random_state is not None:
                np.random.seed(None)

        else:
            i = 0
            for layer in self.layers:
                if hasattr(layer, "bias"):
                    layer.bias = value[i]
                    i += 1

    def get_weight(self) -> list[np.ndarray]:
        """
        Get the weights of the trainable layers of the neural network `Net`.

        Returns
        -------
        value : list[`numpy.ndarray`]
            A list of numpy arrays containing the weights of the trainable layers.

        See Also
        --------
        Net.set_weight : Set the values of the weight in the neural network `Net`.

        Examples
        --------
        >>> from smash.factory import Net
        >>> net = Net()
        >>> net.add_dense(2, input_shape=3, activation="tanh")
        >>> net
        +-------------------------------------------------------+
        | Layer Type         Input/Output Shape  Num Parameters |
        +-------------------------------------------------------+
        | Dense              (3,)/(2,)           8              |
        | Activation (TanH)  (2,)/(2,)           0              |
        +-------------------------------------------------------+
        Total parameters: 8
        Trainable parameters: 8

        Set random weights

        >>> net.set_weight(random_state=0)

        Get the weight values

        >>> net.get_weight()
        [array([[ 0.10694503,  0.47145628,  0.22514328],
                [ 0.09833413, -0.16726395,  0.31963799]])]
        """

        return [layer.weight for layer in self.layers if hasattr(layer, "weight")]

    def get_bias(self) -> list[np.ndarray]:
        """
        Get the biases of the trainable layers of the neural network `Net`.

        Returns
        -------
        value : list[`numpy.ndarray`]
            A list of numpy arrays containing the biases of the trainable layers.

        See Also
        --------
        Net.set_bias : Set the values of the bias in the neural network `Net`.

        Examples
        --------
        >>> from smash.factory import Net
        >>> net = Net()
        >>> net.add_dense(2, input_shape=3, bias_initializer="normal")
        >>> net
        +----------------------------------------------------------+
        | Layer Type            Input/Output Shape  Num Parameters |
        +----------------------------------------------------------+
        | Dense                 (3,)/(2,)           8              |
        +----------------------------------------------------------+
        Total parameters: 8
        Trainable parameters: 8

        Set random biases

        >>> net.set_bias(random_state=0)

        Get the bias values

        >>> net.get_bias()
        [array([[0.01764052, 0.00400157]])]
        """

        return [layer.bias for layer in self.layers if hasattr(layer, "bias")]

    def forward_pass(self, x: np.ndarray):
        """
        Perform a forward pass through the neural network.

        Parameters
        ----------
        x : `numpy.ndarray`
            An array representing the input data for the neural network. The shape of
            this array must be broadcastable into the input shape of the first layer.

        Returns
        -------
        y : `numpy.ndarray`
            The output of the neural network after passing through all layers.

        Examples
        --------
        >>> from smash.factory import Net
        >>> net = Net()
        >>> net.add_dense(12, input_shape=5, activation="tanh")
        >>> net.add_dense(3, activation="softmax")
        >>> net
        +----------------------------------------------------------+
        | Layer Type            Input/Output Shape  Num Parameters |
        +----------------------------------------------------------+
        | Dense                 (5,)/(12,)          72             |
        | Activation (TanH)     (12,)/(12,)         0              |
        | Dense                 (12,)/(3,)          39             |
        | Activation (Softmax)  (3,)/(3,)           0              |
        +----------------------------------------------------------+
        Total parameters: 111
        Trainable parameters: 111

        Set random weights

        >>> net.set_weight(random_state=1)

        Run the forward pass

        >>> import numpy as np
        >>> x = np.array([0.1, 0.11, 0.12, 0.13, 0.14])
        >>> net.forward_pass(x)
        array([[0.31315546, 0.37666753, 0.31017701]])
        """

        x = _standardize_forward_pass_args(self, x)

        return self._forward_pass(x)

    def _forward_pass(self, x: np.ndarray):
        layer_output = x

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
