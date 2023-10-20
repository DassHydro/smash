from __future__ import annotations

from smash._constant import (
    STRUCTURE_NAME,
    SERR_MU_MAPPING_NAME,
    SERR_SIGMA_MAPPING_NAME,
    INPUT_DATA_FORMAT,
    STRUCTURE_RR_PARAMETERS,
    STRUCTURE_RR_STATES,
    SERR_MU_MAPPING_PARAMETERS,
    SERR_SIGMA_MAPPING_PARAMETERS,
    FEASIBLE_RR_PARAMETERS,
    FEASIBLE_RR_INITIAL_STATES,
    FEASIBLE_SERR_MU_PARAMETERS,
    FEASIBLE_SERR_SIGMA_PARAMETERS,
    NN_STATE_SPACE_STRUCTURES,
    WB_INITIALIZER,
)

import pandas as pd
import numpy as np
import os
import warnings
import errno

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash._typing import AnyTuple, Numeric
    from smash.core.model.model import Model
    from smash.fcore._mwd_setup import SetupDT

from smash.factory.samples._standardize import (
    _standardize_generate_samples_random_state,
)


# % TODO: rewrite this standardize with type checking
def _standardize_setup(setup: SetupDT):
    if setup.structure.lower() in STRUCTURE_NAME:
        setup.structure = setup.structure.lower()
    else:
        raise ValueError(
            f"Unknown structure '{setup.structure}'. Choices: {STRUCTURE_NAME}"
        )
    if setup.serr_mu_mapping.capitalize() in SERR_MU_MAPPING_NAME:
        setup.serr_mu_mapping = setup.serr_mu_mapping.capitalize()
    else:
        raise ValueError(
            f"Unknown serr_mu_mapping '{setup.serr_mu_mapping}'. Choices: {SERR_MU_MAPPING_NAME}"
        )

    if setup.serr_sigma_mapping.capitalize() in SERR_SIGMA_MAPPING_NAME:
        setup.serr_sigma_mapping = setup.serr_sigma_mapping.capitalize()
    else:
        raise ValueError(
            f"Unknown serr_sigma_mapping '{setup.serr_sigma_mapping}'. Choices: {SERR_SIGMA_MAPPING_NAME}"
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

    if (et - st).total_seconds() <= 0:
        raise ValueError(
            "argument end_time corresponds to a date earlier or equal to argument start_time"
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

    if setup.structure in NN_STATE_SPACE_STRUCTURES and setup.nhl == 0:
        warnings.warn(
            f"Neural networks are used with no hidden layers in the {setup.structure} structure"
        )


def _standardize_rr_parameters_key(model: Model, key: str) -> str:
    if not isinstance(key, str):
        raise TypeError(f"key argument must be a str")

    if key.lower() not in STRUCTURE_RR_PARAMETERS[model.setup.structure]:
        raise ValueError(
            f"Unknown model rr_parameter '{key}'. Choices: {STRUCTURE_RR_PARAMETERS[model.setup.structure]}"
        )

    return key.lower()


def _standardize_rr_states_key(model: Model, state_kind: str, key: str) -> str:
    if not isinstance(key, str):
        raise TypeError(f"key argument must be a str")

    if key.lower() not in STRUCTURE_RR_STATES[model.setup.structure]:
        raise ValueError(
            f"Unknown model {state_kind} '{key}'. Choices: {STRUCTURE_RR_STATES[model.setup.structure]}"
        )

    return key.lower()


def _standardize_serr_mu_parameters_key(model: Model, key: str) -> str:
    if not isinstance(key, str):
        raise TypeError(f"key argument must be a str")

    if key.lower() not in SERR_MU_MAPPING_PARAMETERS[model.setup.serr_mu_mapping]:
        raise ValueError(
            f"Unknown model serr_mu_parameters '{key}'. Choices: {SERR_MU_MAPPING_PARAMETERS[model.setup.serr_mu_mapping]}"
        )

    return key.lower()


def _standardize_serr_sigma_parameters_key(model: Model, key: str) -> str:
    if not isinstance(key, str):
        raise TypeError(f"key argument must be a str")

    if key.lower() not in SERR_SIGMA_MAPPING_PARAMETERS[model.setup.serr_sigma_mapping]:
        raise ValueError(
            f"Unknown model serr_sigma_parameters '{key}'. Choices: {SERR_SIGMA_MAPPING_PARAMETERS[model.setup.serr_sigma_mapping]}"
        )

    return key.lower()


def _standardize_rr_parameters_value(
    model: Model, key: str, value: Numeric | np.ndarray
) -> Numeric | np.ndarray:
    if not isinstance(value, (int, float, np.ndarray)):
        raise TypeError(
            f"value argument must be of Numeric type (int, float) or np.ndarray"
        )

    arr = np.array(value, ndmin=1)
    l, u = FEASIBLE_RR_PARAMETERS[key]
    l_arr = np.min(arr)
    u_arr = np.max(arr)

    if (
        isinstance(value, np.ndarray)
        and value.shape != model.mesh.flwdir.shape
        and value.size != 1
    ):
        raise ValueError(
            f"Invalid shape for model rr_parameter '{key}'. Could not broadcast input array from shape {value.shape} into shape {model.mesh.flwdir.shape}"
        )

    if l_arr <= l or u_arr >= u:
        raise ValueError(
            f"Invalid value for model rr_parameter '{key}'. Rr_parameter domain [{l_arr}, {u_arr}] is not included in the feasible domain ]{l}, {u}["
        )

    return value


def _standardize_rr_states_value(
    model: Model, state_kind: str, key: str, value: Numeric | np.ndarray
) -> Numeric | np.ndarray:
    if not isinstance(value, (int, float, np.ndarray)):
        raise TypeError(
            f"value argument must be of Numeric type (int, float) or np.ndarray"
        )

    arr = np.array(value, ndmin=1)
    l, u = FEASIBLE_RR_INITIAL_STATES[key]
    l_arr = np.min(arr)
    u_arr = np.max(arr)

    if (
        isinstance(value, np.ndarray)
        and value.shape != model.mesh.flwdir.shape
        and value.size != 1
    ):
        raise ValueError(
            f"Invalid shape for model {state_kind} '{key}'. Could not broadcast input array from shape {value.shape} into shape {model.mesh.flwdir.shape}"
        )

    if l_arr <= l or u_arr >= u:
        raise ValueError(
            f"Invalid value for model {state_kind} '{key}'. {state_kind.capitalize()} domain [{l_arr}, {u_arr}] is not included in the feasible domain ]{l}, {u}["
        )

    return value


def _standardize_serr_mu_parameters_value(
    model: Model, key: str, value: Numeric | np.ndarray
) -> Numeric | np.ndarray:
    if not isinstance(value, (int, float, np.ndarray)):
        raise TypeError(
            f"value argument must be of Numeric type (int, float) or np.ndarray"
        )

    arr = np.array(value, ndmin=1)
    l, u = FEASIBLE_SERR_MU_PARAMETERS[key]
    l_arr = np.min(arr)
    u_arr = np.max(arr)

    if (
        isinstance(value, np.ndarray)
        and value.shape != (model.mesh.ng,)
        and value.size != 1
    ):
        raise ValueError(
            f"Invalid shape for model serr_mu_parameters '{key}'. Could not broadcast input array from shape {value.shape} into shape {(model.mesh.ng,)}"
        )

    if l_arr <= l or u_arr >= u:
        raise ValueError(
            f"Invalid value for model serr_mu_parameters '{key}'. Serr_mu_parameters domain [{l_arr}, {u_arr}] is not included in the feasible domain ]{l}, {u}["
        )

    return value


def _standardize_serr_sigma_parameters_value(
    model: Model, key: str, value: Numeric | np.ndarray
) -> Numeric | np.ndarray:
    if not isinstance(value, (int, float, np.ndarray)):
        raise TypeError(
            f"value argument must be of Numeric type (int, float) or np.ndarray"
        )

    arr = np.array(value, ndmin=1)
    l, u = FEASIBLE_SERR_SIGMA_PARAMETERS[key]
    l_arr = np.min(arr)
    u_arr = np.max(arr)

    if (
        isinstance(value, np.ndarray)
        and value.shape != (model.mesh.ng,)
        and value.size != 1
    ):
        raise ValueError(
            f"Invalid shape for model serr_sigma_parameters '{key}'. Could not broadcast input array from shape {value.shape} into shape {(model.mesh.ng,)}"
        )

    if l_arr <= l or u_arr >= u:
        raise ValueError(
            f"Invalid value for model serr_sigma_parameters '{key}'. Serr_sigma_parameters domain [{l_arr}, {u_arr}] is not included in the feasible domain ]{l}, {u}["
        )

    return value


def _standardize_get_rr_parameters_args(model: Model, key: str) -> str:
    key = _standardize_rr_parameters_key(model, key)

    return key


def _standardize_get_rr_initial_states_args(model: Model, key: str) -> str:
    key = _standardize_rr_states_key(model, "rr_initial_state", key)

    return key


def _standardize_get_serr_mu_parameters_args(model: Model, key: str) -> str:
    key = _standardize_serr_mu_parameters_key(model, key)

    return key


def _standardize_get_serr_sigma_parameters_args(model: Model, key: str) -> str:
    key = _standardize_serr_sigma_parameters_key(model, key)

    return key


def _standardize_get_rr_final_states_args(model: Model, key: str) -> str:
    key = _standardize_rr_states_key(model, "rr_final_state", key)

    return key


def _standardize_set_rr_parameters_args(
    model: Model, key: str, value: Numeric | np.ndarray
) -> AnyTuple:
    key = _standardize_rr_parameters_key(model, key)

    value = _standardize_rr_parameters_value(model, key, value)

    return (key, value)


def _standardize_set_rr_initial_states_args(
    model: Model, key: str, value: Numeric | np.ndarray
) -> AnyTuple:
    state_kind = "rr_initial_state"
    key = _standardize_rr_states_key(model, state_kind, key)

    value = _standardize_rr_states_value(model, state_kind, key, value)

    return (key, value)


def _standardize_set_serr_mu_parameters_args(
    model: Model, key: str, value: Numeric | np.ndarray
) -> AnyTuple:
    key = _standardize_serr_mu_parameters_key(model, key)

    value = _standardize_serr_mu_parameters_value(model, key, value)

    return (key, value)


def _standardize_set_serr_sigma_parameters_args(
    model: Model, key: str, value: Numeric | np.ndarray
) -> AnyTuple:
    key = _standardize_serr_sigma_parameters_key(model, key)

    value = _standardize_serr_sigma_parameters_value(model, key, value)

    return (key, value)


def _standardize_set_nn_parameters_initializer(initializer: str) -> str:
    if initializer not in WB_INITIALIZER:
        raise ValueError(
            f"Unknown initializer: {initializer}. Choices {WB_INITIALIZER}"
        )

    return initializer.lower()


def _standardize_set_nn_parameters_random_state(
    random_state: Numeric | None,
) -> int | None:
    return _standardize_generate_samples_random_state(random_state)


def _standardize_set_nn_parameters_weight_args(
    initializer: str, random_state: Numeric | None
) -> AnyTuple:
    initializer = _standardize_set_nn_parameters_initializer(initializer)
    random_state = _standardize_set_nn_parameters_random_state(random_state)

    return (initializer, random_state)


def _standardize_set_nn_parameters_bias_args(
    initializer: str, random_state: Numeric | None
) -> AnyTuple:
    return _standardize_set_nn_parameters_weight_args(initializer, random_state)
