from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from smash._constant import OPTIMIZABLE_NN_PARAMETERS
from smash.core.simulation.optimize._tools import _get_parameters_b, _net_to_parameters
from smash.fcore._mwd_parameters_manipulation import (
    control_to_parameters as wrap_control_to_parameters,
)
from smash.fcore._mwd_parameters_manipulation import (
    parameters_to_control as wrap_parameters_to_control,
)

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.factory.net.net import Net
    from smash.fcore._mwd_options import OptionsDT, ParametersDT
    from smash.fcore._mwd_returns import ReturnsDT


def _hcost(instance: Model) -> float:
    return instance._output.cost


def _hcost_prime(
    net: Net,
    x: np.ndarray,
    model_params_states: np.ndarray,
    instance: Model,
    parameters: ParametersDT,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
):
    # % Update rr_parameters and/or rr_initial_states using net
    y_reshaped = _net_to_parameters(net, x, model_params_states, parameters)

    # % Change the mapping to trigger distributed control to get distributed gradients
    wrap_options.optimize.mapping = "distributed"

    # % Run adjoint model
    wrap_parameters_to_control(
        instance.setup,
        instance.mesh,
        instance._input_data,
        parameters,
        wrap_options,
    )
    parameters_b = _get_parameters_b(instance, parameters, wrap_options, wrap_returns)

    wrap_control_to_parameters(
        instance.setup, instance.mesh, instance._input_data, parameters_b, wrap_options
    )

    # % Reset mapping to ann
    wrap_options.optimize.mapping = "ann"

    # % Get the gradient of regionalization NN
    grad_reg = []
    for name in model_params_states:
        if name in parameters.rr_parameters.keys:
            ind = np.argwhere(parameters.rr_parameters.keys == name).item()

            grad_reg.append(parameters_b.rr_parameters.values[..., ind])

        elif name in parameters.rr_initial_states.keys:
            ind = np.argwhere(parameters.rr_initial_states.keys == name).item()

            grad_reg.append(parameters_b.rr_initial_states.values[..., ind])

        else:  # nn_parameters excluded from descriptors-to-parameters mapping
            pass

    if y_reshaped:  # in case of Dense (MLP)
        grad_reg = np.reshape(grad_reg, (len(grad_reg), -1)).T

    else:  # in case of CNN
        grad_reg = np.transpose(grad_reg, (1, 2, 0))

    # % Get the gradient of parameterization NN if used
    grad_par = [
        getattr(parameters_b.nn_parameters, key).copy()
        for key in OPTIMIZABLE_NN_PARAMETERS[max(0, instance.setup.n_layers - 1)]
    ]

    return grad_reg, grad_par
