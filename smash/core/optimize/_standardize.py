from __future__ import annotations

from smash.core._constant import (
    ALGORITHM,
    MAPPING,
    STRUCTURE_PARAMETERS,
    STRUCTURE_STATES,
    JOBS_FUN,
    CSIGN_OPTIM,
    ESIGN_OPTIM,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver._mwd_setup import SetupDT
    from smash.solver._mwd_mesh import MeshDT
    from smash.solver._mwd_input_data import Input_DataDT

import numpy as np
import pandas as pd
import warnings


def _standardize_mapping(mapping: str, setup: SetupDT) -> str:

    if isinstance(mapping, str):

        mapping = mapping.lower()

    else:

        raise TypeError(f"mapping argument must be str")

    if mapping not in MAPPING:

        raise ValueError(f"Unknown mapping '{mapping}'. Choices: {MAPPING}")

    if mapping.startswith("hyper") and setup._nd == 0:

        raise ValueError(
            f"'{mapping}' mapping can not be use if no catchment descriptors are available"
        )

    return mapping


def _standardize_algorithm(algorithm: str | None, mapping: str) -> str:

    if algorithm is None:

        if mapping == "uniform":

            algorithm = "sbs"

        elif mapping in ["distributed", "hyper-linear", "hyper-polynomial"]:

            algorithm = "l-bfgs-b"

    else:

        if isinstance(algorithm, str):

            algorithm = algorithm.lower()

        else:

            raise TypeError(f"'algorithm' argument must be str")

        if algorithm not in ALGORITHM:

            raise ValueError(f"Unknown algorithm '{algorithm}'. Choices: {ALGORITHM}")

        if algorithm == "sbs":

            if mapping in ["distributed", "hyper-linear", "hyper-polynomial"]:

                raise ValueError(
                    f"'{algorithm}' algorithm can not be use with '{mapping}' mapping"
                )

        elif algorithm == "nelder-mead":

            if mapping in ["distributed", "hyper-polynomial"]:

                raise ValueError(
                    f"'{algorithm}' algorithm can not be use with '{mapping}' mapping"
                )

        elif algorithm == "l-bfgs-b":

            if mapping == "uniform":

                raise ValueError(
                    f"'{algorithm}' algorithm can not be use with '{mapping}' mapping"
                )

    return algorithm


def _standardize_control_vector(
    control_vector: str | list | tuple | set | None, setup: SetupDT
) -> np.ndarray:

    if control_vector is None:
        control_vector = np.array(STRUCTURE_PARAMETERS[setup.structure])

    else:

        if isinstance(control_vector, str):

            control_vector = np.array(control_vector, ndmin=1)

        elif isinstance(control_vector, set):

            control_vector = np.array(list(control_vector))

        elif isinstance(control_vector, (list, tuple)):

            control_vector = np.array(control_vector)

        else:

            raise TypeError(
                f"'control_vector' argument must be str or list-like object"
            )

        for name in control_vector:

            available = [
                *STRUCTURE_PARAMETERS[setup.structure],
                *STRUCTURE_STATES[setup.structure],
            ]

            if name not in available:

                raise ValueError(
                    f"Unknown parameter or state '{name}' for structure '{setup.structure}' in 'control_vector'. Choices: {available}"
                )

    return control_vector


def _standardize_jobs_fun(
    jobs_fun: str | list | tuple | set, algorithm: str
) -> np.ndarray:

    if isinstance(jobs_fun, str):

        jobs_fun = np.array(jobs_fun, ndmin=1)

    elif isinstance(jobs_fun, set):

        jobs_fun = np.array(list(jobs_fun))

    elif isinstance(jobs_fun, (list, tuple)):

        jobs_fun = np.array(jobs_fun)

    else:
        raise TypeError("jobs_fun argument must be str or list-like object")

    if "kge" in jobs_fun and algorithm == "l-bfgs-b":
        raise ValueError(
            f"'{jobs_fun}' objective function can not be use with '{algorithm}' algorithm (non convex function)"
        )

    list_jobs_fun = JOBS_FUN + CSIGN_OPTIM + ESIGN_OPTIM

    check_obj = np.array([1 if o in list_jobs_fun else 0 for o in jobs_fun])

    if sum(check_obj) < len(check_obj):
        raise ValueError(
            f"Unknown objective function: {np.array(jobs_fun)[np.where(check_obj == 0)]}. Choices {list_jobs_fun}"
        )

    return jobs_fun


def _standardize_wjobs(
    wjobs_fun: list | None, jobs_fun: np.ndarray, algorithm: str
) -> np.ndarray:

    if wjobs_fun is None:

        #% WIP
        if algorithm == "nsga":

            pass

        else:

            wjobs_fun = np.ones(jobs_fun.size) / jobs_fun.size

    else:

        if isinstance(wjobs_fun, set):

            wjobs_fun = np.array(list(wjobs_fun))

        elif isinstance(wjobs_fun, (list, tuple)):

            wjobs_fun = np.array(wjobs_fun)

        else:

            raise TypeError("wjobs_fun argument must list-like object")

        if wjobs_fun.size != jobs_fun.size:

            raise ValueError(
                f"Inconsistent size between jobs_fun ({jobs_fun.size}) and wjobs_fun ({wjobs_fun.size})"
            )

    return wjobs_fun


def _standardize_bounds(
    bounds: list | tuple | set | None, control_vector: np.ndarray, setup: SetupDT
) -> np.ndarray:

    #% Default values
    if bounds is None:

        bounds = np.empty(shape=(control_vector.size, 2), dtype=np.float32)

        for i, name in enumerate(control_vector):

            if name in setup._parameters_name:

                ind = np.argwhere(setup._parameters_name == name)

                bounds[i, :] = (
                    setup._optimize.lb_parameters[ind].item(),
                    setup._optimize.ub_parameters[ind].item(),
                )

            elif name in setup._states_name:

                ind = np.argwhere(setup._states_name == name)

                bounds[i, :] = (
                    setup._optimize.lb_states[ind].item(),
                    setup._optimize.ub_states[ind].item(),
                )

    else:

        if isinstance(bounds, set):

            bounds = np.array(list(bounds))

        elif isinstance(bounds, (list, tuple)):

            bounds = np.array(bounds)

        else:

            raise TypeError(f"bounds argument must be list-like object")

        if bounds.shape[0] != control_vector.size:

            raise ValueError(
                f"Inconsistent size between control_vector ({control_vector.size}) and bounds ({bounds.shape[0]})"
            )

        for i, b in enumerate(bounds):

            if not isinstance(b, (np.ndarray, list, tuple, set)) or len(b) != 2:

                raise ValueError(
                    f"bounds values for '{control_vector[i]}' must be list-like object of length 2"
                )

            if b[0] is None:

                if control_vector[i] in setup._parameters_name:

                    ind = np.argwhere(setup._parameters_name == control_vector[i])
                    b[0] = setup._optimize.lb_parameters[ind].item()

                else:

                    ind = np.argwhere(setup._states_name == control_vector[i])
                    b[0] = setup._optimize.lb_states[ind].item()

            if b[1] is None:

                if control_vector[i] in setup._parameters_name:

                    ind = np.argwhere(setup._parameters_name == control_vector[i])
                    b[1] = setup._optimize.ub_parameters[ind].item()

                else:

                    ind = np.argwhere(setup._states_name == control_vector[i])
                    b[1] = setup._optimize.ub_states[ind].item()

            if b[0] >= b[1]:

                raise ValueError(
                    f"bounds values for '{control_vector[i]}' is invalid lower bound ({b[0]}) is greater than or equal to upper bound ({b[1]})"
                )

    return bounds


def _standardize_gauge(
    gauge: str | list | tuple | set,
    setup: SetupDT,
    mesh: MeshDT,
    input_data: Input_DataDT,
) -> np.ndarray:

    if isinstance(gauge, str):

        if gauge == "all":

            gauge = mesh.code.copy()

        elif gauge == "downstream":

            ind = np.argmax(mesh.area)

            gauge = np.array(mesh.code[ind], ndmin=1)

        elif gauge in mesh.code:

            gauge = np.array(gauge, ndmin=1)

        else:

            raise ValueError(
                f"Unknown gauge alias or code '{gauge}'. Choices: ['all', 'downstream'] or {mesh.code}"
            )

    elif isinstance(gauge, set):

        gauge = np.array(list(gauge))

    elif isinstance(gauge, (list, tuple)):

        gauge = np.array(gauge)

    else:

        raise TypeError(f"gauge argument must be str or list-like object")

    gauge_check = np.array([])

    for i, name in enumerate(gauge):

        if name in mesh.code:

            ind = np.argwhere(mesh.code == name).squeeze()

            if np.all(input_data.qobs[ind, setup._optimize.optimize_start_step :] < 0):

                warnings.warn(
                    f"gauge '{name}' has no available observed discharge. Removed from the optimization"
                )

            else:

                gauge_check = np.append(gauge_check, gauge[i])

        else:

            raise ValueError(f"Unknown gauge code '{name}'. Choices: {mesh.code}")

    if gauge_check.size == 0:

        raise ValueError(
            f"No available observed discharge for optimization at gauge(s) {gauge}"
        )

    else:

        gauge = gauge_check

    return gauge


def _standardize_wgauge(
    wgauge: str | list | tuple | set, gauge: np.ndarray, mesh: MeshDT
) -> np.ndarray:

    weight_arr = np.zeros(shape=mesh.code.size, dtype=np.float32)
    ind = np.in1d(mesh.code, gauge)

    if isinstance(wgauge, str):

        if wgauge == "mean":

            weight_arr[ind] = 1 / gauge.size

        elif wgauge == "area":

            weight_arr[ind] = mesh.area[ind] / sum(mesh.area[ind])

        elif wgauge == "minv_area":

            weight_arr[ind] = (1 / mesh.area[ind]) / sum(1 / mesh.area[ind])

        else:

            raise ValueError(
                f"Unknown wgauge alias '{wgauge}'. Choices: ['mean', 'area', 'minv_area']"
            )

    elif isinstance(wgauge, (list, tuple, set)):

        wgauge = np.array(list(wgauge))

        if wgauge.size != gauge.size:

            raise ValueError(
                f"Inconsistent size between gauge ({gauge.size}) and wgauge ({wgauge.size})"
            )

        else:

            weight_arr[ind] = wgauge

    else:

        raise TypeError(f"wgauge argument must be str or list-like object")

    wgauge = weight_arr

    return wgauge


def _standardize_ost(ost: str | pd.Timestamp | None, setup: SetupDT) -> pd.Timestamp:

    if ost is None:

        ost = pd.Timestamp(setup.start_time)

    else:

        st = pd.Timestamp(setup.start_time)
        et = pd.Timestamp(setup.end_time)

        if isinstance(ost, str):

            try:
                ost = pd.Timestamp(ost)

            except:

                raise ValueError(f"ost '{ost}' argument is an invalid date")

        elif isinstance(ost, pd.Timestamp):

            pass

        else:

            raise TypeError(f"ost argument must be str or pandas.Timestamp object")

        if (ost - st).total_seconds() < 0 or (et - ost).total_seconds() < 0:

            raise ValueError(
                f"ost '{ost}' argument must be between start time '{st}' and end time '{et}'"
            )

    return ost


def _standardize_optimize_args(
    mapping: str,
    algorithm: str | None,
    control_vector: str | list | tuple | set | None,
    jobs_fun: str | list | tuple | set,
    wjobs_fun: list | None,
    bounds: list | tuple | set | None,
    gauge: str | list | tuple | set,
    wgauge: str | list | tuple | set,
    ost: str | pd.Timestamp | None,
    setup: SetupDT,
    mesh: MeshDT,
    input_data: Input_DataDT,
):

    mapping = _standardize_mapping(mapping, setup)

    algorithm = _standardize_algorithm(algorithm, mapping)

    control_vector = _standardize_control_vector(control_vector, setup)

    jobs_fun = _standardize_jobs_fun(jobs_fun, algorithm)

    wjobs_fun = _standardize_wjobs(wjobs_fun, jobs_fun, algorithm)

    bounds = _standardize_bounds(bounds, control_vector, setup)

    gauge = _standardize_gauge(gauge, setup, mesh, input_data)

    wgauge = _standardize_wgauge(wgauge, gauge, mesh)

    ost = _standardize_ost(ost, setup)

    return mapping, algorithm, control_vector, jobs_fun, wjobs_fun, bounds, wgauge, ost


def _standardize_optimize_options(options: dict | None) -> dict:

    if options is None:

        options = {}

    return options


def _standardize_jobs_fun_wo_optimize(
    jobs_fun: str | list | tuple | set,
) -> np.ndarray:

    if isinstance(jobs_fun, str):

        jobs_fun = np.array(jobs_fun, ndmin=1)

    elif isinstance(jobs_fun, set):

        jobs_fun = np.array(list(jobs_fun))

    elif isinstance(jobs_fun, (list, tuple)):

        jobs_fun = np.array(jobs_fun)

    else:
        raise TypeError("jobs_fun argument must be str or list-like object")

    list_jobs_fun = JOBS_FUN + CSIGN_OPTIM + ESIGN_OPTIM

    check_obj = np.array([1 if o in list_jobs_fun else 0 for o in jobs_fun])

    if sum(check_obj) < len(check_obj):
        raise ValueError(
            f"Unknown objective function: {np.array(jobs_fun)[np.where(check_obj == 0)]}. Choices {list_jobs_fun}"
        )

    return jobs_fun


def _standardize_wjobs_wo_optimize(
    wjobs_fun: list | None, jobs_fun: np.ndarray
) -> np.ndarray:

    if wjobs_fun is None:

        wjobs_fun = np.ones(jobs_fun.size) / jobs_fun.size

    else:

        if isinstance(wjobs_fun, set):

            wjobs_fun = np.array(list(wjobs_fun))

        elif isinstance(wjobs_fun, (list, tuple)):

            wjobs_fun = np.array(wjobs_fun)

        else:

            raise TypeError("wjobs_fun argument must list-like object")

        if wjobs_fun.size != jobs_fun.size:

            raise ValueError(
                f"Inconsistent size between jobs_fun ({jobs_fun.size}) and wjobs_fun ({wjobs_fun.size})"
            )

    return wjobs_fun


def _standardize_wo_optimize_args(
    control_vector: str | list | tuple | set | None,
    jobs_fun: str | list | tuple | set,
    wjobs_fun: list | None,
    bounds: list | tuple | set | None,
    gauge: str | list | tuple | set,
    wgauge: str | list | tuple | set,
    ost: str | pd.Timestamp | None,
    setup: SetupDT,
    mesh: MeshDT,
    input_data: Input_DataDT,
):

    control_vector = _standardize_control_vector(control_vector, setup)

    jobs_fun = _standardize_jobs_fun_wo_optimize(jobs_fun)

    wjobs_fun = _standardize_wjobs_wo_optimize(wjobs_fun, jobs_fun)

    bounds = _standardize_bounds(bounds, control_vector, setup)

    gauge = _standardize_gauge(gauge, setup, mesh, input_data)

    wgauge = _standardize_wgauge(wgauge, gauge, mesh)

    ost = _standardize_ost(ost, setup)

    return control_vector, jobs_fun, wjobs_fun, bounds, wgauge, ost


def _standardize_bayes_estimate_args(
    control_vector: str | list | tuple | set | None,
    jobs_fun: str | list | tuple | set,
    wjobs_fun: list | None,
    bounds: list | tuple | set | None,
    gauge: str | list | tuple | set,
    wgauge: str | list | tuple | set,
    ost: str | pd.Timestamp | None,
    setup: SetupDT,
    mesh: MeshDT,
    input_data: Input_DataDT,
):  # add more standardize params (param for build sample, k, .....)!!!!!!!!!!

    control_vector = _standardize_control_vector(control_vector, setup)

    jobs_fun = _standardize_jobs_fun_wo_optimize(jobs_fun)

    wjobs_fun = _standardize_wjobs_wo_optimize(wjobs_fun, jobs_fun)

    bounds = _standardize_bounds(bounds, control_vector, setup)

    gauge = _standardize_gauge(gauge, setup, mesh, input_data)

    wgauge = _standardize_wgauge(wgauge, gauge, mesh)

    ost = _standardize_ost(ost, setup)

    return control_vector, jobs_fun, wjobs_fun, bounds, wgauge, ost
