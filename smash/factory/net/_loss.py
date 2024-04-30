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


def _hcost(instance: Model):
    return instance._output.cost


def _hcost_prime(
    y: np.ndarray,
    parameters: np.ndarray,
    mask: np.ndarray,
    instance: Model,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
):
    # % Set parameters or states
    for i, name in enumerate(parameters):
        if name in instance.rr_parameters.keys:
            ind = np.argwhere(instance.rr_parameters.keys == name).item()

            instance.rr_parameters.values[..., ind][mask] = y[:, i]

        else:
            ind = np.argwhere(instance.rr_initial_states.keys == name).item()

            instance.rr_initial_states.values[..., ind][mask] = y[:, i]

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

    grad = []
    for name in parameters:
        if name in instance.rr_parameters.keys:
            ind = np.argwhere(instance.rr_parameters.keys == name).item()

            grad.append(parameters_b.rr_parameters.values[..., ind][mask])

        else:
            ind = np.argwhere(instance.rr_initial_states.keys == name).item()

            grad.append(parameters_b.rr_initial_states.values[..., ind][mask])

    grad = np.transpose(grad)

    return grad


def _inf_norm(grad: np.ndarray):
    return np.amax(np.abs(grad))
