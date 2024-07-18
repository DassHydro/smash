from __future__ import annotations

from typing import TYPE_CHECKING

from smash._constant import PY_OPTIMIZER, PY_OPTIMIZER_CLASS

# Used inside eval statement
from smash.factory.net._layers import Activation, Dense, Dropout, Scale  # noqa: F401
from smash.factory.net._loss import _hcost, _hcost_prime, _inf_norm
from smash.factory.net._optimizers import SGD, Adagrad, Adam, RMSprop  # noqa: F401
from smash.factory.net._standardize import _standardize_add_args

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

        The graph is set using `Net.add`.

        Examples
        --------
        >>> from smash.factory import Net
        >>> net = Net()
        >>> net.add(layer="dense", options={"input_shape": (6,), "neurons": 32})
        >>> net.add(layer="activation", options={"name": "sigmoid"})
        >>> net.add(layer="dropout", options={"drop_rate": 0.2})

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
        A dictionary saving training information.

        The keys are 'loss_train', 'loss_valid', and 'proj_grad'.
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
            Layer name. Should be one of 'dense', 'activation', 'scale', 'dropout'.

        options : dict
            A dictionary to configure layers added to the network.

            .. hint::
                See options for each layer type:

                - 'dense' :ref:`(see here) <api_reference.sub-packages.net.add_dense>`
                - 'activation' :ref:`(see here) <api_reference.sub-packages.net.add_activation>`
                - 'scale' :ref:`(see here) <api_reference.sub-packages.net.add_scale>`
                - 'dropout' :ref:`(see here) <api_reference.sub-packages.net.add_dropout>`

        Examples
        --------
        Initialize the neural network:

        >>> from smash.factory import Net
        >>> net = Net()

        Define graph:

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

        Display a summary of the network:

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
        """

        layer, options = _standardize_add_args(self, layer, options)

        lay = eval(layer)(**options)

        if not self.layers:  # Check options if first layer
            if "input_shape" in options:
                if not isinstance(options["input_shape"], tuple):
                    raise ValueError(
                        f"input_shape option should be a tuple, not {type(options['input_shape'])}"
                    )

            else:
                raise TypeError("First layer missing required option argument: 'input_shape'")

        else:
            # If be not the first layer then set the input shape to the output shape of the next added layer
            lay._set_input_shape(shape=self.layers[-1].output_shape())

        # Add layer to the network
        self.layers.append(lay)

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

            .. note::
                Dropout, activation, and scaling functions are non-parametric layers,
                meaning they do not have any learnable weights or biases.
                Therefore, it is not necessary to set these layers as trainable
                since they do not involve any weight updates during training.

        Examples
        --------
        >>> net.add(layer="dense", options={"input_shape": (8,), "neurons": 32})
        >>> net.add(layer="activation", options={"name": "relu"})
        >>> net.add(layer="dense", options={"neurons": 16})
        >>> net.add(layer="activation", options={"name": "relu"})
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

        >>> net.set_trainable([True, False, False, False])
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

        if len(trainable) == len(self.layers):
            for i, layer in enumerate(self.layers):
                layer.trainable = trainable[i]

        else:
            raise ValueError(
                f"Inconsistent size between trainable ({len(trainable)}) and the number of layers "
                f"({len(self.layers)})"
            )

    def _compile(
        self,
        optimizer: str,
        learning_param: dict,
        random_state: Numeric | None,
    ):
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

        # % Train model
        for epo in tqdm(range(epochs), desc="    Training"):
            # forward propogation
            y_pred = self._forward_pass(x_train)

            # calculate the gradient of the loss function wrt y_pred
            init_loss_grad = _hcost_prime(y_pred, parameters, mask, instance, wrap_options, wrap_returns)

            # compute loss
            loss = _hcost(instance)
            self.history["loss_train"].append(loss)

            # save optimal weights if early stopping is used
            if early_stopping:
                if epo == 0:
                    loss_opt = {"epo": 0, "value": loss}

                if loss <= loss_opt["value"]:
                    loss_opt["epo"] = epo
                    loss_opt["value"] = loss

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

            # backpropagation and calculate infinity norm of the projected gradient
            loss_grad = self._backward_pass(
                init_loss_grad, inplace=True if epo < epochs - 1 else False
            )  # do not update weights at the last epoch
            self.history["proj_grad"].append(_inf_norm(loss_grad))

            if verbose:
                ret = []

                ret.append(f"{' ' * 4}At epoch")
                ret.append("{:3}".format(epo + 1))
                ret.append("J =" + "{:10.6f}".format(loss))
                ret.append("|proj g| =" + "{:10.6f}".format(self.history["proj_grad"][-1]))

                tqdm.write((" " * 4).join(ret))

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

    def _backward_pass(self, loss_grad: np.ndarray, inplace=True):
        if inplace:
            net = self
        else:
            net = self.copy()

        for layer in reversed(net.layers):
            loss_grad = layer._backward_pass(loss_grad)

        return loss_grad

    def _predict(self, x_train: np.ndarray):
        preds = self._forward_pass(x_train, training=False)

        return preds
