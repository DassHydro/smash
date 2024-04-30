from __future__ import annotations

import datetime
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from smash._constant import (
    DEFAULT_MODEL_SETUP,
    FEASIBLE_RR_INITIAL_STATES,
    FEASIBLE_RR_PARAMETERS,
    FEASIBLE_SERR_MU_PARAMETERS,
    FEASIBLE_SERR_SIGMA_PARAMETERS,
    HYDROLOGICAL_MODULE,
    INPUT_DATA_FORMAT,
    ROUTING_MODULE,
    ROUTING_MODULE_NQZ,
    SERR_MU_MAPPING,
    SERR_MU_MAPPING_PARAMETERS,
    SERR_SIGMA_MAPPING,
    SERR_SIGMA_MAPPING_PARAMETERS,
    SNOW_MODULE,
    STRUCTURE_RR_PARAMETERS,
    STRUCTURE_RR_STATES,
)

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.util._typing import AnyTuple, ListLike, Numeric


def _standardize_model_setup_bool(key: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{key} model setup must be a boolean")

    return value


def _standardize_model_setup_directory(read: bool, key: str, value: str | None) -> str:
    directory_kind = key.split("_")[0]

    if read:
        if value is None:
            raise ValueError(f"{key} model setup must be defined if read_{directory_kind} is set to True")
        elif isinstance(value, str):
            if not os.path.exists(value):
                raise FileNotFoundError(f"No such file or directory '{value}' for {key} model setup")
        else:
            raise TypeError(f"{key} model setup must be a str")
    else:
        value = "..."

    return value


def _standardize_model_setup_format(key: str, value: str) -> str:
    if isinstance(value, str):
        if value.lower() in INPUT_DATA_FORMAT:
            value = value.lower()
        else:
            raise ValueError(
                f"Unkown format '{value}' for {key} in model setup. Choices: {INPUT_DATA_FORMAT}"
            )

    else:
        raise TypeError(f"{key} model setup must be a str")

    return value


def _standardize_model_setup_conversion_factor(key: str, value: Numeric) -> float:
    if isinstance(value, (int, float)):
        value = float(value)
        if value <= 0:
            raise ValueError(f"{key} model setup must be greater than 0")

    else:
        raise TypeError(f"{key} model setup must be of Numeric type (int, float)")

    return value


def _standardize_model_setup_access(key: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{key} model setup must be a str")

    return value


def _standardize_model_setup_snow_module(snow_module: str, **kwargs) -> str:
    if isinstance(snow_module, str):
        if snow_module.lower() in SNOW_MODULE:
            snow_module = snow_module.lower()

        else:
            raise ValueError(
                f"Unknown snow module '{snow_module}' for snow_module in model setup. Choices: {SNOW_MODULE}"
            )

    else:
        raise TypeError("snow_module model setup must be a str")

    return snow_module


def _standardize_model_setup_hydrological_module(hydrological_module: str, **kwargs) -> str:
    if isinstance(hydrological_module, str):
        if hydrological_module.lower() in HYDROLOGICAL_MODULE:
            hydrological_module = hydrological_module.lower()
        else:
            raise ValueError(
                f"Unknown hydrological module '{hydrological_module}' for hydrological_module in model "
                f"setup. Choices: {HYDROLOGICAL_MODULE}"
            )
    else:
        raise TypeError("hydrological_module model setup must be a str")

    return hydrological_module


def _standardize_model_setup_routing_module(routing_module: str, **kwargs) -> str:
    if isinstance(routing_module, str):
        if routing_module.lower() in ROUTING_MODULE:
            routing_module = routing_module.lower()
        else:
            raise ValueError(
                f"Unknown routing module '{routing_module}' for routing_module in model setup. "
                f"Choices: {ROUTING_MODULE}"
            )
    else:
        raise TypeError("routing_module model setup must be a str")

    return routing_module


def _standardize_model_setup_serr_mu_mapping(serr_mu_mapping: str, **kwargs) -> str:
    if isinstance(serr_mu_mapping, str):
        if serr_mu_mapping.capitalize() in SERR_MU_MAPPING:
            serr_mu_mapping = serr_mu_mapping.capitalize()
        else:
            raise ValueError(
                f"Unknown structural error mu mapping '{serr_mu_mapping}' for serr_mu_mapping in model "
                f"setup. Choices: {SERR_MU_MAPPING}"
            )
    else:
        raise TypeError("serr_mu_mapping model setup must be a str")

    return serr_mu_mapping


def _standardize_model_setup_serr_sigma_mapping(serr_sigma_mapping: str, **kwargs) -> str:
    if isinstance(serr_sigma_mapping, str):
        if serr_sigma_mapping.capitalize() in SERR_SIGMA_MAPPING:
            serr_sigma_mapping = serr_sigma_mapping.capitalize()
        else:
            raise ValueError(
                f"Unknown structural error sigma mapping '{serr_sigma_mapping}' for serr_sigma_mapping in "
                f"model setup. Choices: {SERR_SIGMA_MAPPING}"
            )
    else:
        raise TypeError("serr_sigma_mapping model setup must be a str")

    return serr_sigma_mapping


def _standardize_model_setup_dt(dt: Numeric, **kwargs) -> float:
    if isinstance(dt, (int, float)):
        dt = float(dt)
        if dt <= 0:
            raise ValueError("dt model setup must be greater than 0")
    else:
        raise TypeError("dt model setup must be of Numeric type (int, float)")

    return dt


def _standardize_model_setup_start_time(
    start_time: str | datetime.date | pd.Timestamp | None, **kwargs
) -> pd.Timestamp:
    if start_time is None:
        raise ValueError("start_time model setup must be defined")
    elif isinstance(start_time, str):
        try:
            start_time = pd.Timestamp(start_time)
        except Exception:
            raise ValueError(f"start_time '{start_time}' model setup is an invalid date") from None
    elif isinstance(start_time, datetime.date):
        start_time = pd.Timestamp(start_time)

    elif isinstance(start_time, pd.Timestamp):
        pass

    else:
        raise TypeError(
            "start_time model setup must be a str, datetime.date object or pandas.Timestamp object"
        )

    return start_time


def _standardize_model_setup_end_time(
    start_time: pd.Timestamp,
    end_time: str | datetime.date | pd.Timestamp | None,
    **kwargs,
) -> pd.Timestamp:
    if end_time is None:
        raise ValueError("end_time model setup must be defined")
    elif isinstance(end_time, str):
        try:
            end_time = pd.Timestamp(end_time)
        except Exception:
            raise ValueError(f"end_time '{end_time}' model setup is an invalid date") from None
    elif isinstance(end_time, datetime.date):
        end_time = pd.Timestamp(end_time)

    elif isinstance(end_time, pd.Timestamp):
        pass

    else:
        raise TypeError("end_time model setup must be a str, datetime.date object or pandas.Timestamp object")

    # Check that end_time is after start_time
    if (end_time - start_time).total_seconds() <= 0:
        raise ValueError(
            f"end_time model setup '{end_time}' corresponds to a date earlier or equal to start_time model "
            f"setup '{start_time}'"
        )

    return end_time


def _standardize_model_setup_adjust_interception(adjust_interception: bool, **kwrags) -> bool:
    return _standardize_model_setup_bool("adjust_interception", adjust_interception)


def _standardize_model_setup_compute_mean_atmos(compute_mean_atmos: bool, **kwrags) -> bool:
    return _standardize_model_setup_bool("compute_mean_atmos", compute_mean_atmos)


def _standardize_model_setup_read_qobs(read_qobs: bool, **kwrags) -> bool:
    return _standardize_model_setup_bool("read_qobs", read_qobs)


def _standardize_model_setup_qobs_directory(read_qobs: bool, qobs_directory: str | None, **kwargs) -> str:
    return _standardize_model_setup_directory(read_qobs, "qobs_directory", qobs_directory)


def _standardize_model_setup_read_prcp(read_prcp: bool, **kwrags) -> bool:
    return _standardize_model_setup_bool("read_prcp", read_prcp)


def _standardize_model_setup_prcp_format(prcp_format: str, **kwargs) -> str:
    return _standardize_model_setup_format("prcp_format", prcp_format)


def _standardize_model_setup_prcp_conversion_factor(prcp_conversion_factor: str, **kwargs) -> str:
    return _standardize_model_setup_conversion_factor("prcp_conversion_factor", prcp_conversion_factor)


def _standardize_model_setup_prcp_directory(read_prcp: bool, prcp_directory: str | None, **kwargs) -> str:
    return _standardize_model_setup_directory(read_prcp, "prcp_directory", prcp_directory)


def _standardize_model_setup_prcp_access(prcp_access: str, **kwargs) -> str:
    return _standardize_model_setup_access("prcp_access", prcp_access)


def _standardize_model_setup_read_pet(read_pet: bool, **kwrags) -> bool:
    return _standardize_model_setup_bool("read_pet", read_pet)


def _standardize_model_setup_pet_format(pet_format: str, **kwargs) -> str:
    return _standardize_model_setup_format("pet_format", pet_format)


def _standardize_model_setup_pet_conversion_factor(pet_conversion_factor: str, **kwargs) -> str:
    return _standardize_model_setup_conversion_factor("pet_conversion_factor", pet_conversion_factor)


def _standardize_model_setup_pet_directory(read_pet: bool, pet_directory: str | None, **kwargs) -> str:
    return _standardize_model_setup_directory(read_pet, "pet_directory", pet_directory)


def _standardize_model_setup_pet_access(pet_access: str, **kwargs) -> str:
    return _standardize_model_setup_access("pet_access", pet_access)


def _standardize_model_setup_daily_interannual_pet(daily_interannual_pet: bool, **kwargs) -> bool:
    return _standardize_model_setup_bool("daily_interannual_pet", daily_interannual_pet)


def _standardize_model_setup_read_snow(snow_module: str, read_snow: bool, **kwrags) -> bool:
    read_snow = _standardize_model_setup_bool("read_snow", read_snow)

    if read_snow and snow_module == "zero":
        raise ValueError("read_snow model setup can not be set to True if no snow module has been selected")

    return read_snow


def _standardize_model_setup_snow_format(snow_format: str, **kwargs) -> str:
    return _standardize_model_setup_format("snow_format", snow_format)


def _standardize_model_setup_snow_conversion_factor(snow_conversion_factor: str, **kwargs) -> str:
    return _standardize_model_setup_conversion_factor("snow_conversion_factor", snow_conversion_factor)


def _standardize_model_setup_snow_directory(read_snow: bool, snow_directory: str | None, **kwargs) -> str:
    return _standardize_model_setup_directory(read_snow, "snow_directory", snow_directory)


def _standardize_model_setup_snow_access(snow_access: str, **kwargs) -> str:
    return _standardize_model_setup_access("snow_access", snow_access)


def _standardize_model_setup_read_temp(snow_module: str, read_temp: bool, **kwrags) -> bool:
    read_temp = _standardize_model_setup_bool("read_temp", read_temp)

    if read_temp and snow_module == "zero":
        raise ValueError("read_temp model setup can not be set to True if no snow module has been selected")

    return read_temp


def _standardize_model_setup_temp_format(temp_format: str, **kwargs) -> str:
    return _standardize_model_setup_format("temp_format", temp_format)


def _standardize_model_setup_temp_directory(read_temp: bool, temp_directory: str | None, **kwargs) -> str:
    return _standardize_model_setup_directory(read_temp, "temp_directory", temp_directory)


def _standardize_model_setup_temp_access(temp_access: str, **kwargs) -> str:
    return _standardize_model_setup_access("temp_access", temp_access)


def _standardize_model_setup_prcp_partitioning(
    snow_module: str,
    read_snow: bool,
    read_temp: bool,
    prcp_partitioning: bool,
    **kwargs,
) -> bool:
    prcp_partitioning = _standardize_model_setup_bool("prcp_partitioning", prcp_partitioning)

    if prcp_partitioning and snow_module == "zero":
        raise ValueError(
            "prcp_partitioning model setup can not be set to True if no snow module has been selected "
            "(snow_module is set to 'zero')"
        )

    if prcp_partitioning and not read_temp:
        raise ValueError(
            "prcp_partitioning model setup can not be set to True if no temperature data is read "
            "(read_temp is set to False)"
        )

    if prcp_partitioning and read_snow:
        warnings.warn(
            "prcp_partitioning and read_snow model setup are set to True. The snow data read will be summed "
            "with the precipitation and re-partitioned",
            stacklevel=2,
        )

    return prcp_partitioning


def _standardize_model_setup_sparse_storage(sparse_storage: bool, **kwargs) -> bool:
    return _standardize_model_setup_bool("sparse_storage", sparse_storage)


def _standardize_model_setup_read_descriptor(read_descriptor: bool, **kwrags) -> bool:
    return _standardize_model_setup_bool("read_descriptor", read_descriptor)


def _standardize_model_setup_descriptor_format(descriptor_format: str, **kwargs) -> str:
    return _standardize_model_setup_format("descriptor_format", descriptor_format)


def _standardize_model_setup_descriptor_directory(
    read_descriptor: bool, descriptor_directory: str | None, **kwargs
) -> str:
    return _standardize_model_setup_directory(read_descriptor, "descriptor_directory", descriptor_directory)


def _standardize_model_setup_descriptor_name(descriptor_name: ListLike | None, **kwargs) -> np.ndarray:
    if descriptor_name is None:
        descriptor_name = np.empty(shape=0)
    elif isinstance(descriptor_name, (list, tuple, np.ndarray)):
        descriptor_name = np.array(descriptor_name, ndmin=1)
    else:
        raise TypeError("descriptor_name model setup must be of ListLike type (List, Tuple, np.ndarray)")

    return descriptor_name


def _standardize_model_setup(setup: dict) -> dict:
    if isinstance(setup, dict):
        pop_keys = []
        for key in setup.keys():
            if key not in DEFAULT_MODEL_SETUP.keys():
                pop_keys.append(key)
                warnings.warn(
                    f"Unknown model setup key '{key}'. Choices: {list(DEFAULT_MODEL_SETUP.keys())}",
                    stacklevel=2,
                )

        for key in pop_keys:
            setup.pop(key)

    else:
        raise TypeError("setup model argument must be a dictionary")

    for key, value in DEFAULT_MODEL_SETUP.items():
        setup.setdefault(key, value)
        func = eval(f"_standardize_model_setup_{key}")
        setup[key] = func(**setup)

    return setup


def _standardize_model_setup_finalize(setup: dict):
    setup["structure"] = "-".join(
        [
            setup["snow_module"],
            setup["hydrological_module"],
            setup["routing_module"],
        ]
    )

    setup["snow_module_present"] = setup["snow_module"] != "zero"

    setup["ntime_step"] = int((setup["end_time"] - setup["start_time"]).total_seconds() / setup["dt"])
    setup["nd"] = setup["descriptor_name"].size
    setup["nrrp"] = len(STRUCTURE_RR_PARAMETERS[setup["structure"]])
    setup["nrrs"] = len(STRUCTURE_RR_STATES[setup["structure"]])
    setup["nsep_mu"] = len(SERR_MU_MAPPING_PARAMETERS[setup["serr_mu_mapping"]])
    setup["nsep_sigma"] = len(SERR_SIGMA_MAPPING_PARAMETERS[setup["serr_sigma_mapping"]])
    setup["nqz"] = ROUTING_MODULE_NQZ[setup["routing_module"]]

    setup["start_time"] = setup["start_time"].strftime("%Y-%m-%d %H:%M")
    setup["end_time"] = setup["end_time"].strftime("%Y-%m-%d %H:%M")


# % We assume that users use smash.factory.generate_mesh to generate the mesh
# % Otherwise, good luck !
def _standardize_model_mesh(mesh: dict) -> dict:
    if not isinstance(mesh, dict):
        raise TypeError("mesh model argument must be a dictionary")
    return mesh


def _standardize_model_args(setup: dict, mesh: dict) -> AnyTuple:
    setup = _standardize_model_setup(setup)

    _standardize_model_setup_finalize(setup)

    mesh = _standardize_model_mesh(mesh)

    return (setup, mesh)


def _standardize_rr_parameters_key(model: Model, key: str) -> str:
    if not isinstance(key, str):
        raise TypeError("key argument must be a str")

    if key.lower() not in STRUCTURE_RR_PARAMETERS[model.setup.structure]:
        raise ValueError(
            f"Unknown model rr_parameter '{key}'. Choices: {STRUCTURE_RR_PARAMETERS[model.setup.structure]}"
        )

    return key.lower()


def _standardize_rr_states_key(model: Model, state_kind: str, key: str) -> str:
    if not isinstance(key, str):
        raise TypeError("key argument must be a str")

    if key.lower() not in STRUCTURE_RR_STATES[model.setup.structure]:
        raise ValueError(
            f"Unknown model {state_kind} '{key}'. Choices: {STRUCTURE_RR_STATES[model.setup.structure]}"
        )

    return key.lower()


def _standardize_serr_mu_parameters_key(model: Model, key: str) -> str:
    if not isinstance(key, str):
        raise TypeError("key argument must be a str")

    if key.lower() not in SERR_MU_MAPPING_PARAMETERS[model.setup.serr_mu_mapping]:
        raise ValueError(
            f"Unknown model serr_mu_parameters '{key}'. Choices: "
            f"{SERR_MU_MAPPING_PARAMETERS[model.setup.serr_mu_mapping]}"
        )

    return key.lower()


def _standardize_serr_sigma_parameters_key(model: Model, key: str) -> str:
    if not isinstance(key, str):
        raise TypeError("key argument must be a str")

    if key.lower() not in SERR_SIGMA_MAPPING_PARAMETERS[model.setup.serr_sigma_mapping]:
        raise ValueError(
            f"Unknown model serr_sigma_parameters '{key}'. Choices: "
            f"{SERR_SIGMA_MAPPING_PARAMETERS[model.setup.serr_sigma_mapping]}"
        )

    return key.lower()


def _standardize_rr_parameters_value(
    model: Model, key: str, value: Numeric | np.ndarray
) -> Numeric | np.ndarray:
    if not isinstance(value, (int, float, np.ndarray)):
        raise TypeError("value argument must be of Numeric type (int, float) or np.ndarray")

    arr = np.array(value, ndmin=1)
    low, upp = FEASIBLE_RR_PARAMETERS[key]
    low_arr = np.min(arr)
    upp_arr = np.max(arr)

    if isinstance(value, np.ndarray) and value.shape != model.mesh.flwdir.shape and value.size != 1:
        raise ValueError(
            f"Invalid shape for model rr_parameter '{key}'. Could not broadcast input array from shape "
            f"{value.shape} into shape {model.mesh.flwdir.shape}"
        )

    if low_arr <= low or upp_arr >= upp:
        raise ValueError(
            f"Invalid value for model rr_parameter '{key}'. rr_parameter domain [{low_arr}, {upp_arr}] is "
            f"not included in the feasible domain ]{low}, {upp}["
        )

    return value


def _standardize_rr_states_value(
    model: Model, state_kind: str, key: str, value: Numeric | np.ndarray
) -> Numeric | np.ndarray:
    if not isinstance(value, (int, float, np.ndarray)):
        raise TypeError("value argument must be of Numeric type (int, float) or np.ndarray")

    arr = np.array(value, ndmin=1)
    low, upp = FEASIBLE_RR_INITIAL_STATES[key]
    low_arr = np.min(arr)
    upp_arr = np.max(arr)

    if isinstance(value, np.ndarray) and value.shape != model.mesh.flwdir.shape and value.size != 1:
        raise ValueError(
            f"Invalid shape for model {state_kind} '{key}'. Could not broadcast input array from shape "
            f"{value.shape} into shape {model.mesh.flwdir.shape}"
        )

    if low_arr <= low or upp_arr >= upp:
        raise ValueError(
            f"Invalid value for model {state_kind} '{key}'. {state_kind} domain [{low_arr}, {upp_arr}] is "
            f"not included in the feasible domain ]{low}, {upp}["
        )

    return value


def _standardize_serr_mu_parameters_value(
    model: Model, key: str, value: Numeric | np.ndarray
) -> Numeric | np.ndarray:
    if not isinstance(value, (int, float, np.ndarray)):
        raise TypeError("value argument must be of Numeric type (int, float) or np.ndarray")

    arr = np.array(value, ndmin=1)
    low, upp = FEASIBLE_SERR_MU_PARAMETERS[key]
    low_arr = np.min(arr)
    upp_arr = np.max(arr)

    if isinstance(value, np.ndarray) and value.shape != (model.mesh.ng,) and value.size != 1:
        raise ValueError(
            f"Invalid shape for model serr_mu_parameter '{key}'. Could not broadcast input array from shape "
            f"{value.shape} into shape {(model.mesh.ng,)}"
        )

    if low_arr <= low or upp_arr >= upp:
        raise ValueError(
            f"Invalid value for model serr_mu_parameter '{key}'. serr_mu_parameter domain "
            f"[{low_arr}, {upp_arr}] is not included in the feasible domain ]{low}, {upp}["
        )

    return value


def _standardize_serr_sigma_parameters_value(
    model: Model, key: str, value: Numeric | np.ndarray
) -> Numeric | np.ndarray:
    if not isinstance(value, (int, float, np.ndarray)):
        raise TypeError("value argument must be of Numeric type (int, float) or np.ndarray")

    arr = np.array(value, ndmin=1)
    low, upp = FEASIBLE_SERR_SIGMA_PARAMETERS[key]
    low_arr = np.min(arr)
    upp_arr = np.max(arr)

    if isinstance(value, np.ndarray) and value.shape != (model.mesh.ng,) and value.size != 1:
        raise ValueError(
            f"Invalid shape for model serr_sigma_parameter '{key}'. Could not broadcast input array from "
            f"shape {value.shape} into shape {(model.mesh.ng,)}"
        )

    if low_arr <= low or upp_arr >= upp:
        raise ValueError(
            f"Invalid value for model serr_sigma_parameter '{key}'. serr_sigma_parameter domain "
            f"[{low_arr}, {upp_arr}] is not included in the feasible domain ]{low}, {upp}["
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


def _standardize_set_rr_parameters_args(model: Model, key: str, value: Numeric | np.ndarray) -> AnyTuple:
    key = _standardize_rr_parameters_key(model, key)

    value = _standardize_rr_parameters_value(model, key, value)

    return (key, value)


def _standardize_set_rr_initial_states_args(model: Model, key: str, value: Numeric | np.ndarray) -> AnyTuple:
    state_kind = "rr_initial_state"
    key = _standardize_rr_states_key(model, state_kind, key)

    value = _standardize_rr_states_value(model, state_kind, key, value)

    return (key, value)


def _standardize_set_serr_mu_parameters_args(model: Model, key: str, value: Numeric | np.ndarray) -> AnyTuple:
    key = _standardize_serr_mu_parameters_key(model, key)

    value = _standardize_serr_mu_parameters_value(model, key, value)

    return (key, value)


def _standardize_set_serr_sigma_parameters_args(
    model: Model, key: str, value: Numeric | np.ndarray
) -> AnyTuple:
    key = _standardize_serr_sigma_parameters_key(model, key)

    value = _standardize_serr_sigma_parameters_value(model, key, value)

    return (key, value)
