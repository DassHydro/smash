from __future__ import annotations

from smash._constant import FEASIBLE_OPR_PARAMETERS, FEASIBLE_OPR_INITIAL_STATES

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.fcore._mwd_parameters import ParametersDT

import numpy as np


def _standardize_opr_parameter_state(parameter: ParametersDT):
    # % standardize parameters
    for param_name in FEASIBLE_OPR_PARAMETERS:
        param_array = getattr(parameter.opr_parameters, param_name)

        low, upp = FEASIBLE_OPR_PARAMETERS[param_name]

        if np.logical_or(param_array <= low, param_array >= upp).any():
            raise ValueError(
                f"Invalid value for model parameter {param_name}. Feasible domain: ({low}, {upp})"
            )

    # % standardize states
    for state_name in FEASIBLE_OPR_INITIAL_STATES:
        state_array = getattr(parameter.opr_initial_states, state_name)

        low, upp = FEASIBLE_OPR_INITIAL_STATES[state_name]

        if np.logical_or(state_array <= low, state_array >= upp).any():
            raise ValueError(
                f"Invalid value for model state {state_name}. Feasible domain: ({low}, {upp})"
            )
