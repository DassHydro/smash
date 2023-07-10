from __future__ import annotations

from smash._constant import (
    STRUCTURE_NAME,
    INPUT_DATA_FORMAT,
    FEASIBLE_OPR_PARAMETERS,
    BOUNDS_OPR_STATES,
    BOUNDS_OPR_PARAMETERS,
    TOL_BOUNDS,
)

from smash._typing import numeric

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver._mwd_setup import SetupDT

import pandas as pd
import numpy as np
import os
import warnings
import errno


def _standardize_setup(setup: SetupDT):
    setup.structure = setup.structure.lower()

    if setup.structure not in STRUCTURE_NAME:
        raise ValueError(
            f"Unknown structure '{setup.structure}'. Choices: {STRUCTURE_NAME}"
        )

    if setup.dt < 0:
        raise ValueError("argument dt is lower than 0")

    if not setup.dt in [900, 3_600, 86_400]:
        warnings.warn(
            "argument dt is not set to a classical value (900, 3600, 86400 seconds)",
            UserWarning,
        )

    if setup.start_time == "...":
        raise ValueError("argument start_time is not defined")

    if setup.end_time == "...":
        raise ValueError("argument end_time is not defined")

    try:
        st = pd.Timestamp(setup.start_time)
    except:
        raise ValueError("argument start_time is not a valid date")

    try:
        et = pd.Timestamp(setup.end_time)
    except:
        raise ValueError("argument end_time is not a valid date")

    if (et - st).total_seconds() < 0:
        raise ValueError(
            "argument end_time corresponds to an earlier date than start_time"
        )

    if setup.read_qobs and setup.qobs_directory == "...":
        raise ValueError("argument read_qobs is True and qobs_directory is not defined")

    if setup.read_qobs and not os.path.exists(setup.qobs_directory):
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            setup.qobs_directory,
        )

    if setup.read_prcp and setup.prcp_directory == "...":
        raise ValueError("argument read_prcp is True and prcp_directory is not defined")

    if setup.read_prcp and not os.path.exists(setup.prcp_directory):
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            setup.prcp_directory,
        )

    if setup.prcp_format not in INPUT_DATA_FORMAT:
        raise ValueError(
            f"Unknown prcp_format '{setup.prcp_format}'. Choices: {INPUT_DATA_FORMAT}"
        )

    if setup.prcp_conversion_factor < 0:
        raise ValueError("argument prcp_conversion_factor is lower than 0")

    if setup.read_pet and setup.pet_directory == "...":
        raise ValueError("argument read_pet is True and pet_directory is not defined")

    if setup.read_pet and not os.path.exists(setup.pet_directory):
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            setup.pet_directory,
        )

    if setup.pet_format not in INPUT_DATA_FORMAT:
        raise ValueError(
            f"Unknown pet_format '{setup.pet_format}'. Choices: {INPUT_DATA_FORMAT}"
        )

    if setup.pet_conversion_factor < 0:
        raise ValueError("argument pet_conversion_factor is lower than 0")

    if setup.read_descriptor and setup.descriptor_directory == "...":
        raise ValueError(
            "argument read_descriptor is True and descriptor_directory is not defined"
        )

    if setup.read_descriptor and not os.path.exists(setup.descriptor_directory):
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            setup.descriptor_directory,
        )

    if setup.read_descriptor and setup.nd == 0:
        raise ValueError(
            "argument read_descriptor is True and descriptor_name is not defined"
        )

    if setup.descriptor_format not in INPUT_DATA_FORMAT:
        raise ValueError(
            f"Unknown descriptor_format '{setup.descriptor_format}'. Choices: {INPUT_DATA_FORMAT}"
        )


def _standardize_parameters(parameters: dict, mesh_shape: tuple):
    if isinstance(parameters, dict):
        ret_param = {}

        for key, value in parameters.items():
            key = key.lower()

            if key in BOUNDS_OPR_PARAMETERS:
                if isinstance(value, (numeric, np.ndarray)):
                    if isinstance(value, np.ndarray) and value.shape != mesh_shape:
                        raise ValueError(
                            f"Model parameter {key} must be of the same shape as the mesh ({value.shape} != {mesh_shape})"
                        )

                    low, upp = BOUNDS_OPR_PARAMETERS[key]

                    low_feas, upp_feas = FEASIBLE_OPR_PARAMETERS[key]

                    if np.logical_or(value <= low_feas, value >= upp_feas).any():
                        raise ValueError(
                            f"Invalid value for Model parameter {key}. Feasible domain: ({low_feas}, {upp_feas})"
                        )

                    else:
                        if np.logical_or(
                            value < (low - TOL_BOUNDS), value > (upp + TOL_BOUNDS)
                        ).any():
                            warnings.warn(
                                f"Model parameter {key} is out of default boundary condition [{low}, {upp}]"
                            )

                    ret_param[key] = value

                else:
                    raise TypeError(
                        f"The value of model parameter {key} must be of numeric type or np.ndarray"
                    )

            else:
                warnings.warn(
                    f"Unknown model parameter {key}. Choice: {list(BOUNDS_OPR_PARAMETERS)}"
                )

        return ret_param

    else:
        raise TypeError("param_states argument must be a dictionary")


def _standardize_states(states: dict, mesh_shape: tuple):
    if isinstance(states, dict):
        ret_states = {}

        for key, value in states.items():
            key = key.lower()

            if key in BOUNDS_OPR_STATES:
                if isinstance(value, (numeric, np.ndarray)):
                    if isinstance(value, np.ndarray) and value.shape != mesh_shape:
                        raise ValueError(
                            f"Initial state {key} must be of the same shape as the mesh ({value.shape} != {mesh_shape})"
                        )

                    low, upp = BOUNDS_OPR_STATES[key]

                    if np.logical_or(
                        value < (low - TOL_BOUNDS), value > (upp + TOL_BOUNDS)
                    ).any():
                        warnings.warn(
                            f"Initial state {key} is out of default range [{low}, {upp}]"
                        )

                    ret_states[key] = value

                else:
                    raise TypeError(
                        f"The value of initial state {key} must be of numeric type or np.ndarray"
                    )

            else:
                warnings.warn(
                    f"Unknown model state {key}. Choice: {list(BOUNDS_OPR_STATES)}"
                )

        return ret_states

    else:
        raise TypeError("param_states argument must be a dictionary")


def _standardize_parameter_name(parameter: str):
    if isinstance(parameter, str):
        parameter = parameter.lower()

        if parameter in BOUNDS_OPR_PARAMETERS:
            return parameter

        else:
            raise ValueError(
                f"Unknown model parameter {parameter}. Choice: {list(BOUNDS_OPR_PARAMETERS)}"
            )

    else:
        raise TypeError(f"parameter must be a str")


def _standardize_state_name(state: str):
    if isinstance(state, str):
        state = state.lower()

        if state in BOUNDS_OPR_STATES:
            return state

        else:
            raise ValueError(
                f"Unknown model state {state}. Choice: {list(BOUNDS_OPR_STATES)}"
            )

    else:
        raise TypeError(f"state must be a str")
