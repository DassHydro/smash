from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from smash.fcore._mw_forward import forward_run_b as wrap_forward_run_b
from smash.fcore._mwd_parameters_manipulation import (
    control_to_parameters as wrap_control_to_parameters,
)
from smash.fcore._mwd_parameters_manipulation import (
    parameters_to_control as wrap_parameters_to_control,
)

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.fcore._mwd_options import OptionsDT
    from smash.fcore._mwd_returns import ReturnsDT
    from smash.util._typing import AnyTuple


def _hcost(instance: Model) -> float:
    return instance._output.cost


def _hcost_prime(
    y: np.ndarray,
    parameters: np.ndarray,
    instance: Model,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
) -> AnyTuple:
    if y.ndim < 3:
        y = y.reshape(instance.mesh.flwdir.shape + (-1,))
        return_3d_grad = False
    else:
        return_3d_grad = True

    # % Set parameters or states
    for i, name in enumerate(parameters):
        if name in instance.rr_parameters.keys:
            ind = np.argwhere(instance.rr_parameters.keys == name).item()

            instance.rr_parameters.values[..., ind] = y[..., i]

        else:
            ind = np.argwhere(instance.rr_initial_states.keys == name).item()

            instance.rr_inital_states.values[..., ind] = y[..., i]

    # % Run adjoint model
    wrap_parameters_to_control(
        instance.setup,
        instance.mesh,
        instance._input_data,
        instance._parameters,
        wrap_options,
    )

    parameters_b = instance._parameters.copy()
    output_b = instance._output.copy()
    output_b.cost = np.float32(1)

    wrap_forward_run_b(
        instance.setup,
        instance.mesh,
        instance._input_data,
        instance._parameters,
        parameters_b,
        instance._output,
        output_b,
        wrap_options,
        wrap_returns,
    )

    wrap_control_to_parameters(
        instance.setup, instance.mesh, instance._input_data, parameters_b, wrap_options
    )

    # % Get the gradient of NN for regionalization
    grad = []
    for name in parameters:
        if name in instance.rr_parameters.keys:
            ind = np.argwhere(instance.rr_parameters.keys == name).item()

            grad.append(parameters_b.rr_parameters.values[..., ind])

        else:
            ind = np.argwhere(instance.rr_initial_states.keys == name).item()

            grad.append(parameters_b.rr_initial_states.values[..., ind])

    if return_3d_grad:
        grad = np.transpose(grad, (1, 2, 0))
    else:
        grad = np.reshape(grad, (len(grad), -1)).T

    # % Get the gradient of NN in the forward hydrological model if used
    grad_h = [
        {"weight": layer.weight.copy(), "bias": layer.bias.copy()}
        for layer in parameters_b.nn_parameters.layers
    ]

    return grad, grad_h


def _inf_norm(grad: np.ndarray) -> float:
    return np.amax(np.abs(grad))
