from __future__ import annotations

from smash._constant import (
    OPR_PARAMETERS,
    OPR_STATES,
    LOW_FEASIBLE_OPR_PARAMETERS,
    LOW_OPR_INITIAL_STATES,
    UPP_OPR_INITIAL_STATES,
    LOW_OPTIM_OPR_PARAMETERS,
    UPP_OPTIM_OPR_PARAMETERS,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver._mwd_parameters import ParametersDT

import numpy as np
import warnings


def _standardize_paramstates(paramstate: ParametersDT, tol=1e-9):
    # % standardize parameters
    for i, param_name in enumerate(OPR_PARAMETERS):
        param_array = getattr(paramstate.opr_parameters, param_name)

        low_feas = LOW_FEASIBLE_OPR_PARAMETERS[i]

        low = LOW_OPTIM_OPR_PARAMETERS[i]
        upp = UPP_OPTIM_OPR_PARAMETERS[i]

        if np.any(param_array <= low_feas):
            raise ValueError(
                f"Invalid value for Model parameter {param_name} (<= {low_feas})"
            )

        else:
            if np.logical_or(
                param_array < (low - tol), param_array > (upp + tol)
            ).any():
                warnings.warn(
                    f"Model parameter {param_name} is out of default boundary condition [{low}, {upp}]"
                )

    # % standardize states
    for i, state_name in enumerate(OPR_STATES):
        state_array = getattr(paramstate.opr_initial_states, state_name)

        low = LOW_OPR_INITIAL_STATES[i]
        upp = UPP_OPR_INITIAL_STATES[i]

        if np.logical_or(state_array < (low - tol), state_array > (upp + tol)).any():
            warnings.warn(
                f"Initial state {state_name} is out of default range [{low}, {upp}]"
            )
