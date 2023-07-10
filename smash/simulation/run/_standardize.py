from __future__ import annotations

from smash._constant import (
    OPR_PARAMETERS,
    OPR_STATES,
    LOW_FEASIBLE_OPR_PARAMETERS,
    BOUNDS_OPR_INITIAL_STATES,
    BOUNDS_OPR_PARAMETERS,
    TOL_PARAMSTATES,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver._mwd_parameters import ParametersDT

import numpy as np
import warnings


def _standardize_paramstates(paramstate: ParametersDT):
    # % standardize parameters
    for param_name in OPR_PARAMETERS:
        param_array = getattr(paramstate.opr_parameters, param_name)

        low_feas = LOW_FEASIBLE_OPR_PARAMETERS[param_name]

        low, upp = BOUNDS_OPR_PARAMETERS[param_name]

        if np.any(param_array <= low_feas):
            raise ValueError(
                f"Invalid value for Model parameter {param_name} (<= {low_feas})"
            )

        else:
            if np.logical_or(
                param_array < (low - TOL_PARAMSTATES), param_array > (upp + TOL_PARAMSTATES)
            ).any():
                warnings.warn(
                    f"Model parameter {param_name} is out of default boundary condition [{low}, {upp}]"
                )

    # % standardize states
    for state_name in OPR_STATES:
        state_array = getattr(paramstate.opr_initial_states, state_name)

        low, upp = BOUNDS_OPR_INITIAL_STATES[state_name]

        if np.logical_or(state_array < (low - TOL_PARAMSTATES), state_array > (upp + TOL_PARAMSTATES)).any():
            warnings.warn(
                f"Initial state {state_name} is out of default range [{low}, {upp}]"
            )
