from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from smash._constant import OPTIMIZABLE_NN_PARAMETERS
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


def _hcost(instance: Model) -> float:
    return instance._output.cost


def _hcost_prime(
    y: np.ndarray,
    parameters: np.ndarray,
    instance: Model,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
):
    if y.ndim < 3:
        y = y.reshape(instance.mesh.flwdir.shape + (-1,))
        return_3d_grad = False
    else:  # in case of CNN
        return_3d_grad = True

    # % Set parameters or states
    for i, name in enumerate(parameters):
        if name in instance.rr_parameters.keys:
            ind = np.argwhere(instance.rr_parameters.keys == name).item()

            instance.rr_parameters.values[..., ind] = y[..., i]

        elif name in instance.rr_initial_states.keys:
            ind = np.argwhere(instance.rr_initial_states.keys == name).item()

            instance.rr_initial_states.values[..., ind] = y[..., i]

        else:  # nn_parameters excluded from descriptors-to-parameters mapping
            pass

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

    # % Get the gradient of regionalization NN
    grad_reg = []
    for name in parameters:
        if name in instance.rr_parameters.keys:
            ind = np.argwhere(instance.rr_parameters.keys == name).item()

            grad_reg.append(parameters_b.rr_parameters.values[..., ind])

        elif name in instance.rr_initial_states.keys:
            ind = np.argwhere(instance.rr_initial_states.keys == name).item()

            grad_reg.append(parameters_b.rr_initial_states.values[..., ind])

        else:  # nn_parameters excluded from descriptors-to-parameters mapping
            pass

    if return_3d_grad:  # in case of CNN
        grad_reg = np.transpose(grad_reg, (1, 2, 0))
    else:
        grad_reg = np.reshape(grad_reg, (len(grad_reg), -1)).T

    # % Get the gradient of parameterization NN if used
    grad_par = [
        getattr(parameters_b.nn_parameters, key).copy()
        for key in OPTIMIZABLE_NN_PARAMETERS[max(0, instance.setup.n_layers - 1)]
    ]

    return grad_reg, grad_par


def _inf_norm(grad: np.ndarray | list) -> float:
    if isinstance(grad, list):
        if grad:  # If not an empty list
            return max(_inf_norm(g) for g in grad)

        else:
            return 0

    elif isinstance(grad, np.ndarray):
        if grad.size > 0:
            return np.amax(np.abs(grad))

        else:
            return 0

    else:  # Should be unreachable
        pass
