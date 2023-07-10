from __future__ import annotations

from smash._constant import (
    FEASIBLE_OPR_PARAMETERS,
    BOUNDS_OPR_STATES,
    BOUNDS_OPR_PARAMETERS,
    TOL_BOUNDS,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver._mwd_parameters import ParametersDT

import numpy as np
import warnings


def _standardize_parameter(parameter: ParametersDT):
    # % standardize parameters
    for param_name in BOUNDS_OPR_PARAMETERS:
        param_array = getattr(parameter.opr_parameters, param_name)

        low, upp = BOUNDS_OPR_PARAMETERS[param_name]

        low_feas, upp_feas = FEASIBLE_OPR_PARAMETERS[param_name]

        if np.logical_or(param_array <= low_feas, param_array >= upp_feas).any():
            raise ValueError(
                f"Invalid value for Model parameter {param_name}. Feasible domain: ({low_feas}, {upp_feas})"
            )

        else:
            if np.logical_or(
                param_array < (low - TOL_BOUNDS), param_array > (upp + TOL_BOUNDS)
            ).any():
                warnings.warn(
                    f"Model parameter {param_name} is out of default boundary condition [{low}, {upp}]"
                )

    # % standardize states
    for state_name in BOUNDS_OPR_STATES:
        state_array = getattr(parameter.opr_initial_states, state_name)

        low, upp = BOUNDS_OPR_STATES[state_name]

        if np.logical_or(
            state_array < (low - TOL_BOUNDS), state_array > (upp + TOL_BOUNDS)
        ).any():
            warnings.warn(
                f"Initial state {state_name} is out of default range [{low}, {upp}]"
            )
