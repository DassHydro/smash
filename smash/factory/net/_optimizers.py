from __future__ import annotations

import numpy as np


class SGD:
    """
    Compile the neural network with Stochastic Gradient Descent (SGD) optimizer.

    Options
    -------
    learning_rate : float, default 0.01
        The learning rate used to update the weights during training.

    momentum : float, default 0
        The momentum used to smooth the gradient updates.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0, **unknown_options):
        self.learning_rate = learning_rate

        self.momentum = momentum
        self.w_updt = None

    def update(self, w, grad_wrt_w):
        if self.w_updt is None:
            self.w_updt = np.zeros(np.shape(w))

        # Use momentum if set
        self.w_updt = self.momentum * self.w_updt + (1 - self.momentum) * grad_wrt_w

        return w - self.learning_rate * self.w_updt


class Adam:
    """
    Compile the neural network with Adaptive Moment Estimation (Adam) optimizer.

    Options
    -------
    learning_rate : float, default 0.001
        The learning rate used to update the weights during training.

    b1 : float, default 0.9
        Exponential decay rate for the first moment estimate.

    b2 : float, default 0.999
        Exponential decay rate for the second moment estimate.
    """

    # TODO: Add function check_unknown_options
    def __init__(
        self,
        learning_rate: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps=1e-8,
        **unknown_options,
    ):
        self.learning_rate = learning_rate

        self.eps = eps
        self.m = None
        self.v = None

        # Decay rates
        self.b1 = b1
        self.b2 = b2

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray):
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
    """
    Compile the neural network with Adaptive Gradient (Adagrad) optimizer.

    Options
    -------
    learning_rate : float, default 0.01
        The learning rate used to update the weights during training.

    """

    # TODO: Add function check_unknown_options
    def __init__(self, learning_rate: float = 0.01, eps=1e-8, **unknown_options):
        self.learning_rate = learning_rate

        self.G = None  # Sum of squares of the gradients
        self.eps = eps

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray):
        if self.G is None:
            self.G = np.zeros(np.shape(w))

        # Add the square of the gradient of the loss function at w
        self.G += np.power(grad_wrt_w, 2)

        # Adaptive gradient with higher learning rate for sparse data
        return w - self.learning_rate * grad_wrt_w / np.sqrt(self.G + self.eps)


class RMSprop:
    """
    Compile the neural network with Root Mean Square Propagation (RMSprop) optimizer.

    Options
    -------
    learning_rate : float, default 0.001
        The learning rate used to update the weights during training.

    rho : float, default 0.9
        The decay rate for the running average of the squared gradients.
    """

    # TODO: Add function check_unknown_options
    def __init__(
        self,
        learning_rate: float = 0.001,
        rho: float = 0.9,
        eps=1e-8,
        **unknown_options,
    ):
        self.learning_rate = learning_rate

        self.Eg = None  # Running average of the square gradients at w
        self.eps = eps
        self.rho = rho

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray):
        if self.Eg is None:
            self.Eg = np.zeros(np.shape(grad_wrt_w))

        self.Eg = self.rho * self.Eg + (1 - self.rho) * np.power(grad_wrt_w, 2)

        # Divide the learning rate for a weight by a running average of the magnitudes of recent gradients for
        # that weight
        return w - self.learning_rate * grad_wrt_w / np.sqrt(self.Eg + self.eps)
