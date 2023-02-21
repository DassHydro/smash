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

from smash.core._event_segmentation import _standardize_event_seg_options

from smash.solver._mw_derived_type_update import update_optimize_setup

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver._mwd_setup import SetupDT
    from smash.solver._mwd_mesh import MeshDT
    from smash.solver._mwd_input_data import Input_DataDT

import numpy as np
import pandas as pd
import warnings


def _standardize_mapping(mapping: str) -> str:
    if isinstance(mapping, str):
        mapping = mapping.lower()

    else:
        raise TypeError(f"mapping argument must be str")

    if mapping not in MAPPING:
        raise ValueError(f"Unknown mapping '{mapping}'. Choices: {MAPPING}")

    return mapping


def _standardize_descriptors(input_data: Input_DataDT, setup: SetupDT) -> None:
    # For the moment, return None
    # TODO: add warnings for rejecting uniform descriptors

    if setup._nd == 0:
        raise ValueError(
            f"The optimization method chosen can not be used if no catchment descriptors are available"
        )

    else:
        for i in range(setup._nd):
            d = input_data.descriptor[..., i]

            if np.all(d == d[0, 0]):
                raise ValueError(
                    f"Cannot optimize the Model with spatially uniform descriptor {setup.descriptor_name[i]}"
                )


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
        # % WIP
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
    user_bounds: dict | None, control_vector: np.ndarray, setup: SetupDT
) -> np.ndarray:
    
    #% Default values
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
    
    
    # % Default values
    if user_bounds is None:
        
        pass
        
    else:
        
        if isinstance(user_bounds, dict):
            
            for name,b in user_bounds.items():
                
                if name in control_vector:
                
                    if not isinstance(b, (np.ndarray, list, tuple, set)) or len(b) != 2:
                        
                        raise ValueError(
                            f"bounds values for '{name}' must be list-like object of length 2"
                        )
                    
                    if not ( type(b[0])==float or type(b[0])==np.float64 or type(b[0])==type(None) ):
                        
                        raise ValueError(f"bounds value for '{name}' must be a type of float, np.float64 or None")
                        
                    if not ( type(b[1])==float or type(b[1])==np.float64 or type(b[1])==type(None) ):
                        
                        raise ValueError(f"bounds value for '{name}' must be a type of float, np.float64 or None")
                    
                    ind = np.argwhere(control_vector == name)
                    
                    if b[0] is not None:
                        
                        bounds[ind, 0] = user_bounds[name][0]
                    
                    if b[1] is not None:
                        
                        bounds[ind, 1] = user_bounds[name][1]
                    
                    if bounds[ind, 0] >= bounds[ind, 1]:
                        
                        raise ValueError(
                            f"bounds values for '{name}' is invalid, lower bound ({bounds[ind, 0]}) is greater than or equal to upper bound ({bounds[ind, 1]})"
                        )
                
                else :
                    
                    raise ValueError(
                            f"bounds values for '{name}' cannot be changed because '{name}' is not present in the control vector)"
                        )
                
        else:
            raise TypeError(f"bounds argument must be dict-like object")
    
    
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

        elif wgauge == "median":
            weight_arr[ind] = -50

        elif wgauge == "area":
            weight_arr[ind] = mesh.area[ind] / sum(mesh.area[ind])

        elif wgauge == "minv_area":
            weight_arr[ind] = (1 / mesh.area[ind]) / sum(1 / mesh.area[ind])

        else:
            raise ValueError(
                f"Unknown wgauge alias '{wgauge}'. Choices: ['mean', 'median', 'area', 'minv_area']"
            )

    elif isinstance(wgauge, (list, tuple, set)):
        wgauge = np.array(list(wgauge))

        if wgauge.size != gauge.size:
            raise ValueError(
                f"Inconsistent size between gauge ({gauge.size}) and wgauge ({wgauge.size})"
            )

        elif np.any(wgauge < 0):
            raise ValueError(f"Weight gauge can not receive negative values ({wgauge})")

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


def _standardize_optimize_options(options: dict | None) -> dict:
    if options is None:
        options = {}

    return options


def _standardize_jobs_fun_wo_mapping(
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


def _standardize_wjobs_wo_mapping(
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


def _standardize_bayes_k(
    k: int | float | range | list | tuple | set | np.ndarray,
):
    if isinstance(k, (int, float, list)):
        pass

    elif isinstance(k, (range, np.ndarray, tuple, set)):
        k = list(k)

    else:
        raise TypeError("k argument must be numerical or list-like object")

    return k


def _standardize_optimize_args(
    mapping: str,
    algorithm: str | None,
    control_vector: str | list | tuple | set | None,
    jobs_fun: str | list | tuple | set,
    wjobs_fun: list | None,
    event_seg: dict | None,
    bounds: list | tuple | set | None,
    gauge: str | list | tuple | set,
    wgauge: str | list | tuple | set,
    ost: str | pd.Timestamp | None,
    setup: SetupDT,
    mesh: MeshDT,
    input_data: Input_DataDT,
):
    mapping = _standardize_mapping(mapping)

    if mapping.startswith("hyper"):
        _standardize_descriptors(input_data, setup)

    algorithm = _standardize_algorithm(algorithm, mapping)

    control_vector = _standardize_control_vector(control_vector, setup)

    jobs_fun = _standardize_jobs_fun(jobs_fun, algorithm)

    wjobs_fun = _standardize_wjobs(wjobs_fun, jobs_fun, algorithm)

    event_seg = _standardize_event_seg_options(event_seg)

    # % Update optimize setup derived type according to new optimize args.
    # % This Fortran subroutine reset optimize_setup values and realloc arrays.
    # % After wjobs_fun to realloc considering new size.
    # % Before bounds to be consistent with default Fortran bounds.
    update_optimize_setup(
        setup._optimize,
        setup._ntime_step,
        setup._nd,
        mesh.ng,
        mapping,
        jobs_fun.size,
    )

    bounds = _standardize_bounds(bounds, control_vector, setup)

    gauge = _standardize_gauge(gauge, setup, mesh, input_data)

    wgauge = _standardize_wgauge(wgauge, gauge, mesh)

    ost = _standardize_ost(ost, setup)

    return (
        mapping,
        algorithm,
        control_vector,
        jobs_fun,
        wjobs_fun,
        event_seg,
        bounds,
        wgauge,
        ost,
    )


def _standardize_bayes_estimate_args(
    control_vector: str | list | tuple | set | None,
    jobs_fun: str | list | tuple | set,
    wjobs_fun: list | None,
    event_seg: dict | None,
    bounds: list | tuple | set | None,
    gauge: str | list | tuple | set,
    wgauge: str | list | tuple | set,
    ost: str | pd.Timestamp | None,
    setup: SetupDT,
    mesh: MeshDT,
    input_data: Input_DataDT,
    k: int | float | range | list | tuple | set | np.ndarray,
):
    control_vector = _standardize_control_vector(control_vector, setup)

    jobs_fun = _standardize_jobs_fun_wo_mapping(jobs_fun)

    wjobs_fun = _standardize_wjobs_wo_mapping(wjobs_fun, jobs_fun)

    event_seg = _standardize_event_seg_options(event_seg)

    # % Update optimize setup derived type according to new optimize args.
    # % This Fortran subroutine reset optimize_setup values and realloc arrays.
    # % After wjobs_fun to realloc considering new size.
    # % Before bounds to be consistent with default Fortran bounds.
    update_optimize_setup(
        setup._optimize,
        setup._ntime_step,
        setup._nd,
        mesh.ng,
        "...",
        jobs_fun.size,
    )

    bounds = _standardize_bounds(bounds, control_vector, setup)

    gauge = _standardize_gauge(gauge, setup, mesh, input_data)

    wgauge = _standardize_wgauge(wgauge, gauge, mesh)

    ost = _standardize_ost(ost, setup)

    k = _standardize_bayes_k(k)

    return control_vector, jobs_fun, wjobs_fun, event_seg, bounds, wgauge, ost, k


def _standardize_bayes_optimize_args(
    mapping: str,
    algorithm: str | None,
    control_vector: str | list | tuple | set | None,
    jobs_fun: str | list | tuple | set,
    wjobs_fun: list | None,
    event_seg: dict | None,
    bounds: list | tuple | set | None,
    gauge: str | list | tuple | set,
    wgauge: str | list | tuple | set,
    ost: str | pd.Timestamp | None,
    setup: SetupDT,
    mesh: MeshDT,
    input_data: Input_DataDT,
    k: int | float | range | list | tuple | set | np.ndarray,
):
    mapping = _standardize_mapping(mapping)

    if mapping.startswith("hyper"):
        _standardize_descriptors(input_data, setup)

    algorithm = _standardize_algorithm(algorithm, mapping)

    control_vector = _standardize_control_vector(control_vector, setup)

    jobs_fun = _standardize_jobs_fun(jobs_fun, algorithm)

    wjobs_fun = _standardize_wjobs(wjobs_fun, jobs_fun, algorithm)

    event_seg = _standardize_event_seg_options(event_seg)

    # % Update optimize setup derived type according to new optimize args.
    # % This Fortran subroutine reset optimize_setup values and realloc arrays.
    # % After wjobs_fun to realloc considering new size.
    # % Before bounds to be consistent with default Fortran bounds.
    update_optimize_setup(
        setup._optimize,
        setup._ntime_step,
        setup._nd,
        mesh.ng,
        mapping,
        jobs_fun.size,
    )

    bounds = _standardize_bounds(bounds, control_vector, setup)

    gauge = _standardize_gauge(gauge, setup, mesh, input_data)

    wgauge = _standardize_wgauge(wgauge, gauge, mesh)

    ost = _standardize_ost(ost, setup)

    k = _standardize_bayes_k(k)

    return (
        mapping,
        algorithm,
        control_vector,
        jobs_fun,
        wjobs_fun,
        event_seg,
        bounds,
        wgauge,
        ost,
        k,
    )


def _standardize_ann_optimize_args(
    control_vector: str | list | tuple | set | None,
    jobs_fun: str | list | tuple | set,
    wjobs_fun: list | None,
    event_seg: dict | None,
    bounds: list | tuple | set | None,
    gauge: str | list | tuple | set,
    wgauge: str | list | tuple | set,
    ost: str | pd.Timestamp | None,
    setup: SetupDT,
    mesh: MeshDT,
    input_data: Input_DataDT,
):
    _standardize_descriptors(input_data, setup)

    control_vector = _standardize_control_vector(control_vector, setup)

    jobs_fun = _standardize_jobs_fun_wo_mapping(jobs_fun)

    wjobs_fun = _standardize_wjobs_wo_mapping(wjobs_fun, jobs_fun)

    event_seg = _standardize_event_seg_options(event_seg)

    # % Update optimize setup derived type according to new optimize args.
    # % This Fortran subroutine reset optimize_setup values and realloc arrays.
    # % After wjobs_fun to realloc considering new size.
    # % Before bounds to be consistent with default Fortran bounds.
    update_optimize_setup(
        setup._optimize,
        setup._ntime_step,
        setup._nd,
        mesh.ng,
        "...",
        jobs_fun.size,
    )

    bounds = _standardize_bounds(bounds, control_vector, setup)

    gauge = _standardize_gauge(gauge, setup, mesh, input_data)

    wgauge = _standardize_wgauge(wgauge, gauge, mesh)

    ost = _standardize_ost(ost, setup)

    return control_vector, jobs_fun, wjobs_fun, event_seg, bounds, wgauge, ost
