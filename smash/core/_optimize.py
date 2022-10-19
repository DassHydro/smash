from __future__ import annotations

from smash.solver._mwd_common import name_parameters, name_states
from smash.solver._mwd_setup import SetupDT
from smash.solver._mw_run import forward_run
from smash.solver._mw_optimize import (
    optimize_sbs,
    optimize_lbfgsb,
    hyper_optimize_lbfgsb,
)
from smash.solver._mwd_cost import compute_jobs

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model
    from smash.solver._mwd_mesh import MeshDT
    from smash.solver._mwd_input_data import Input_DataDT
    from smash.solver._mwd_parameters import ParametersDT
    from smash.solver._mwd_states import StatesDT
    from smash.solver._mwd_output import OutputDT

import warnings
import numpy as np
import scipy.optimize
import pandas as pd

ALGORITHM = ["sbs", "l-bfgs-b", "nelder-mead"]

JOBS_FUN = [
    "nse",
    "kge",
    "kge2",
    "se",
    "rmse",
    "logarithmique",
]

MAPPING = ["uniform", "distributed", "hyper-linear", "hyper-polynomial"]


def _optimize_sbs(
    instance: Model,
    control_vector: list[str],
    jobs_fun: str,
    mapping: str,
    bounds: list[float],
    wgauge: list[float],
    ost: pd.Timestamp,
    maxiter: int = 100,
    **unknown_options,
):

    _check_unknown_options(unknown_options)

    instance.setup._algorithm = "sbs"

    setup_bgd = SetupDT(instance.setup._nd)

    instance.setup._optim_parameters = 0
    instance.setup._optim_states = 0
    instance.setup._ub_parameters = setup_bgd._ub_parameters.copy()
    instance.setup._lb_parameters = setup_bgd._lb_parameters.copy()
    instance.setup._lb_states = setup_bgd._lb_states.copy()
    instance.setup._ub_states = setup_bgd._ub_states.copy()

    for i, name in enumerate(control_vector):

        if name in name_parameters:

            ind = np.argwhere(name_parameters == name)

            instance.setup._optim_parameters[ind] = 1

            instance.setup._lb_parameters[ind] = bounds[i][0]
            instance.setup._ub_parameters[ind] = bounds[i][1]

        #% Already check, must be states if not parameters
        else:

            ind = np.argwhere(name_states == name)

            instance.setup._optim_states[ind] = 1

            instance.setup._lb_states[ind] = bounds[i][0]
            instance.setup._ub_states[ind] = bounds[i][1]

    instance.setup._jobs_fun = jobs_fun

    instance.setup._mapping = mapping

    instance.mesh._wgauge = wgauge

    st = pd.Timestamp(instance.setup.start_time.decode().strip())

    instance.setup._optim_start_step = (
        ost - st
    ).total_seconds() / instance.setup.dt + 1

    instance.setup._maxiter = maxiter

    _optimize_message(instance, control_vector, mapping)

    optimize_sbs(
        instance.setup,
        instance.mesh,
        instance.input_data,
        instance.parameters,
        instance.states,
        instance.output,
    )


def _optimize_lbfgsb(
    instance: Model,
    control_vector: list[str],
    jobs_fun: str,
    mapping: str,
    bounds: list[float],
    wgauge: list[float],
    ost: pd.Timestamp,
    maxiter: int = 100,
    jreg_fun: str = "prior",
    wjreg: float = 0.0,
    **unknown_options,
):

    _check_unknown_options(unknown_options)

    instance.setup._algorithm = "l-bfgs-b"

    setup_bgd = SetupDT(instance.setup._nd)

    #% Set default values
    instance.setup._optim_parameters = 0
    instance.setup._optim_states = 0
    instance.setup._ub_parameters = setup_bgd._ub_parameters.copy()
    instance.setup._lb_parameters = setup_bgd._lb_parameters.copy()
    instance.setup._lb_states = setup_bgd._lb_states.copy()
    instance.setup._ub_states = setup_bgd._ub_states.copy()

    for i, name in enumerate(control_vector):

        if name in name_parameters:

            ind = np.argwhere(name_parameters == name)

            instance.setup._optim_parameters[ind] = 1

            instance.setup._lb_parameters[ind] = bounds[i][0]
            instance.setup._ub_parameters[ind] = bounds[i][1]

        #% Already check, must be states if not parameters
        else:

            ind = np.argwhere(name_states == name)

            instance.setup._optim_states[ind] = 1

            instance.setup._lb_states[ind] = bounds[i][0]
            instance.setup._ub_states[ind] = bounds[i][1]

    instance.setup._jobs_fun = jobs_fun

    instance.setup._mapping = mapping

    instance.mesh._wgauge = wgauge

    st = pd.Timestamp(instance.setup.start_time.decode().strip())

    instance.setup._optim_start_step = (
        ost - st
    ).total_seconds() / instance.setup.dt + 1

    instance.setup._maxiter = maxiter
    instance.setup._jreg_fun = jreg_fun
    instance.setup._wjreg = wjreg

    _optimize_message(instance, control_vector, mapping)

    if mapping.startswith("hyper"):

        hyper_optimize_lbfgsb(
            instance.setup,
            instance.mesh,
            instance.input_data,
            instance.parameters,
            instance.states,
            instance.output,
        )

    else:

        optimize_lbfgsb(
            instance.setup,
            instance.mesh,
            instance.input_data,
            instance.parameters,
            instance.states,
            instance.output,
        )


def _optimize_nelder_mead(
    instance: Model,
    control_vector: list[str],
    jobs_fun: str,
    mapping: str,
    bounds: list[float],
    wgauge: list[float],
    ost: pd.Timestamp,
    maxiter=None,
    maxfev=None,
    disp=False,
    return_all=False,
    initial_simplex=None,
    xatol=0.0001,
    fatol=0.0001,
    adaptive=True,
    **unknown_options,
):
    global callback_args

    _check_unknown_options(unknown_options)

    instance.setup._algorithm = "nelder-mead"

    instance.setup._jobs_fun = jobs_fun

    instance.setup._mapping = mapping

    instance.mesh._wgauge = wgauge

    st = pd.Timestamp(instance.setup.start_time.decode().strip())

    instance.setup._optim_start_step = (
        ost - st
    ).total_seconds() / instance.setup.dt + 1

    _optimize_message(instance, control_vector, mapping)

    callback_args = {"iterate": 0, "nfg": 0, "J": 0}

    if mapping == "uniform":

        parameters_bgd = instance.parameters.copy()
        states_bgd = instance.states.copy()

        x = _parameters_states_to_x(instance, control_vector)

        callback_args["nfg"] += 1
        callback_args["J"] = instance.output.cost

        _callback(x)

        res = scipy.optimize.minimize(
            _problem,
            x,
            args=(instance, control_vector, parameters_bgd, states_bgd),
            bounds=bounds,
            method="nelder-mead",
            callback=_callback,
            options={
                "maxiter": maxiter,
                "maxfev": maxfev,
                "disp": disp,
                "return_all": return_all,
                "initial_simplex": initial_simplex,
                "xatol": xatol,
                "fatol": fatol,
                "adaptive": adaptive,
            },
        )

        _problem(res.x, instance, control_vector, parameters_bgd, states_bgd)

    #! WIP
    elif mapping == "hyper-linear":

        #! Change for hyper_ regularization
        parameters_bgd = instance.parameters.copy()
        states_bgd = instance.states.copy()

        x = _hyper_parameters_states_to_x(instance, control_vector)

        _hyper_problem(x, instance, control_vector, parameters_bgd, states_bgd, bounds)

        _callback(x)

        res = scipy.optimize.minimize(
            _hyper_problem,
            x,
            args=(instance, control_vector, parameters_bgd, states_bgd, bounds),
            method="nelder-mead",
            callback=_callback,
            options={
                "maxiter": maxiter,
                "maxfev": maxfev,
                "disp": disp,
                "return_all": return_all,
                "initial_simplex": initial_simplex,
                "xatol": xatol,
                "fatol": fatol,
                "adaptive": adaptive,
            },
        )

        _hyper_problem(
            res.x, instance, control_vector, parameters_bgd, states_bgd, bounds
        )

    _callback(res.x)

    if res.success:

        print(f"{' ' * 4}CONVERGENCE: (XATOL, FATOL) < ({xatol}, {fatol})")

    else:

        print(f"{' ' * 4}STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT")


def _optimize_message(instance: Model, control_vector: list[str], mapping: str):

    sp4 = " " * 4

    algorithm = instance.setup._algorithm.decode().strip()
    jobs_fun = instance.setup._jobs_fun.decode().strip()
    jreg_fun = instance.setup._jreg_fun.decode().strip()
    wjreg = instance.setup._wjreg
    parameters = [el for el in control_vector if el in name_parameters]
    states = [el for el in control_vector if el in name_states]
    code = [
        el
        for ind, el in enumerate(instance.mesh.code.tobytes("F").decode().split())
        if instance.mesh._wgauge[ind] > 0
    ]
    gauge = ["{:.6f}".format(el) for el in instance.mesh._wgauge if el > 0]

    if mapping == "uniform":
        mapping_eq = "k(X)"
        len_parameters = len(parameters)
        len_states = len(states)
        nx = 1

    elif mapping == "distributed":
        mapping_eq = "k(x)"
        len_parameters = len(parameters)
        len_states = len(states)
        nx = instance.mesh.nac

    elif mapping == "hyper-linear":
        mapping_eq = "k(x) = a0 + a1 * D1 + ... + an * Dn"
        len_parameters = len(parameters) * (1 + instance.setup._nd)
        len_states = len(states) * (1 + instance.setup._nd)
        nx = 1

    elif mapping == "hyper-polynomial":
        mapping_eq = "k(x) = a0 + a1 * D1 ** b1 + ... + an * Dn ** bn"
        len_parameters = len(parameters) * (1 + 2 * instance.setup._nd)
        len_states = len(states) * (1 + 2 * instance.setup._nd)
        nx = 1

    ret = []

    ret.append("</> Optimize Model J")
    ret.append(f"Algorithm: '{algorithm}'")
    ret.append(f"Jobs function: '{jobs_fun}'")

    if algorithm == "l-bfgs-b":
        ret.append(f"Jreg function: '{jreg_fun}'")
        ret.append(f"wJreg: {'{:.6f}'.format(wjreg)}")

    ret.append(f"Mapping: '{mapping}' {mapping_eq}")
    ret.append(f"Nx: {nx}")
    ret.append(f"Np: {len_parameters} [ {' '.join(parameters)} ]")
    ret.append(f"Ns: {len_states} [ {' '.join(states)} ]")
    ret.append(f"Ng: {len(code)} [ {' '.join(code)} ]")
    ret.append(f"wg: {len(gauge)} [ {' '.join(gauge)} ]")

    print(f"\n{sp4}".join(ret) + "\n")


def _parameters_states_to_x(instance: Model, control_vector: list[str]) -> np.ndarray:

    ac_ind = np.unravel_index(
        np.argmax(instance.mesh.active_cell, axis=None), instance.mesh.active_cell.shape
    )

    x = np.zeros(shape=len(control_vector), dtype=np.float32)

    for ind, name in enumerate(control_vector):

        if name in name_parameters:

            x[ind] = getattr(instance.parameters, name)[ac_ind]

        else:

            x[ind] = getattr(instance.states, name)[ac_ind]

    return x


def _hyper_parameters_states_to_x(
    instance: Model, control_vector: list[str]
) -> np.ndarray:

    ac_ind = np.unravel_index(
        np.argmax(instance.mesh.active_cell, axis=None), instance.mesh.active_cell.shape
    )

    nd_step = 1 + instance.setup._nd

    x = np.zeros(shape=len(control_vector) * nd_step, dtype=np.float32)

    for ind, name in enumerate(control_vector):

        if name in name_parameters:

            x[ind * nd_step] = getattr(instance.parameters, name)[ac_ind]

        else:

            x[ind * nd_step] = getattr(instance.states, name)[ac_ind]

    return x


def _x_to_parameters_states(x: np.ndarray, instance: Model, control_vector: list[str]):

    for ind, name in enumerate(control_vector):

        if name in name_parameters:

            setattr(
                instance.parameters,
                name,
                np.where(
                    instance.mesh.active_cell == 1,
                    x[ind],
                    getattr(instance.parameters, name),
                ),
            )

        else:

            setattr(
                instance.states,
                name,
                np.where(
                    instance.mesh.active_cell == 1,
                    x[ind],
                    getattr(instance.states, name),
                ),
            )


def _x_to_hyper_parameters_states(
    x: np.ndarray, instance: Model, control_vector: list[str], bounds: list[tuple]
):

    nd_step = 1 + instance.setup._nd

    for ind, name in enumerate(control_vector):

        lb, ub = bounds[ind]

        value = x[ind * nd_step] * np.ones(
            shape=(instance.mesh.nrow, instance.mesh.ncol), dtype=np.float32
        )

        for i in range(instance.setup._nd):

            value += x[ind * nd_step + (i + 1)] * instance.input_data.descriptor[..., i]

        #% Hard coded bounds check (WIP)
        value = np.where(value < lb, lb, value)
        value = np.where(value > ub, ub, value)

        if name in name_parameters:

            setattr(instance.parameters, name, value)

        else:

            setattr(instance.states, name, value)


def _problem(
    x: np.ndarray,
    instance: Model,
    control_vector: list[str],
    parameters_bgd: ParametersDT,
    states_bgd: StatesDT,
):

    global callback_args

    _x_to_parameters_states(x, instance, control_vector)

    forward_run(
        instance.setup,
        instance.mesh,
        instance.input_data,
        instance.parameters,
        parameters_bgd,
        instance.states,
        states_bgd,
        instance.output,
        False,
    )

    callback_args["nfg"] += 1
    callback_args["J"] = instance.output.cost

    return instance.output.cost


def _hyper_problem(
    x: np.ndarray,
    instance: Model,
    control_vector: list[str],
    parameters_bgd: ParametersDT,
    states_bgd: StatesDT,
    bounds: list[tuple],
):

    global callback_args

    _x_to_hyper_parameters_states(x, instance, control_vector, bounds)

    forward_run(
        instance.setup,
        instance.mesh,
        instance.input_data,
        instance.parameters,
        parameters_bgd,
        instance.states,
        states_bgd,
        instance.output,
        False,
    )

    callback_args["nfg"] += 1
    callback_args["J"] = instance.output.cost

    return instance.output.cost


def _callback(x: np.ndarray):

    global callback_args

    sp4 = " " * 4

    ret = []

    ret.append(f"{sp4}At iterate")
    ret.append("{:3}".format(callback_args["iterate"]))
    ret.append("nfg =" + "{:5}".format(callback_args["nfg"]))
    ret.append("J =" + "{:10.6f}".format(callback_args["J"]))

    callback_args["iterate"] += 1

    print(sp4.join(ret))


def _standardize_algorithm(algorithm: str) -> str:

    if isinstance(algorithm, str):

        if algorithm.lower() in ALGORITHM:

            algorithm = algorithm.lower()

        else:

            raise ValueError(f"Unknown algorithm '{algorithm}'. Choices: {ALGORITHM}")

    else:

        raise TypeError(f"'algorithm' argument must be str")

    return algorithm


def _standardize_control_vector(control_vector: list, algorithm: str) -> list:

    if isinstance(control_vector, (list, tuple, set)):

        control_vector = list(control_vector)

        for name in control_vector:

            if not name in [*name_parameters, *name_states]:

                raise ValueError(
                    f"Unknown parameter or state '{name}' in 'control_vector'. Choices: {[*name_parameters, *name_states]}"
                )

                raise ValueError(
                    f"state optimization not available with '{algorithm}' algorithm"
                )
    else:

        raise TypeError(f"'control_vector' argument must be list-like object")

    return control_vector


def _standardize_jobs_fun(jobs_fun: str, algorithm: str) -> str:

    if isinstance(jobs_fun, str):

        if not jobs_fun in JOBS_FUN:

            raise ValueError(
                f"Unknown objective function '{jobs_fun}'. Choices: {JOBS_FUN}"
            )

        elif jobs_fun in ["kge"] and algorithm == "l-bfgs-b":

            raise ValueError(
                f"'{jobs_fun}' objective function can not be use with 'l-bfgs-b' algorithm (non convex function)"
            )

    else:

        raise TypeError(f"'jobs_fun' argument must be str")

    return jobs_fun


def _standardize_mapping(mapping: str, algorithm: str, setup: SetupDT) -> str:

    if mapping:

        mapping = mapping.lower()

        if not mapping in MAPPING:

            raise ValueError(f"Unknown mapping '{mapping}'. Choices: {MAPPING}")

        if algorithm == "sbs":

            if mapping in ["distributed", "hyper-linear", "hyper-polynomial"]:

                raise ValueError(
                    f"'{mapping}' mapping can not be use with '{algorithm}' algorithm"
                )

        elif algorithm == "nelder-mead":

            if mapping in ["distributed", "hyper-polynomial"]:

                raise ValueError(
                    f"'{mapping}' mapping can not be use with '{algorithm}' algorithm"
                )

            elif mapping == "hyper-linear" and setup._nd == 0:

                raise ValueError(
                    f"'{mapping}' mapping can not be use if no catchment descriptor are available"
                )

        elif algorithm == "l-bfgs-b":

            if mapping == "uniform":

                raise ValueError(
                    f"'{mapping}' mapping can not be use with '{algorithm}' algorithm"
                )

            elif mapping in ["hyper-linear", "hyper-polynomial"] and setup._nd == 0:

                raise ValueError(
                    f"'{mapping}' mapping can not be use if no catchment descriptor are available"
                )

    #% Default values
    else:

        if algorithm in ["sbs", "nelder-mead"]:

            mapping = "uniform"

        elif algorithm == "l-bfgs-b":

            mapping = "distributed"

    return mapping


def _standardize_bounds(
    bounds: (list, tuple, set), control_vector: str, setup: SetupDT
) -> list:

    if bounds:

        if isinstance(bounds, (list, tuple, set)):

            bounds = list(bounds)

            if len(bounds) != len(control_vector):

                raise ValueError(
                    f"Inconsistent size between 'control_vector' ({len(control_vector)}) and 'bounds' ({len(bounds)})"
                )

            else:

                for i, lb_ub in enumerate(bounds):

                    if not isinstance(lb_ub, (list, tuple, set)) or len(lb_ub) != 2:

                        raise ValueError(
                            f"'bounds' values for '{control_vector[i]}' must be list-like object of length 2"
                        )

                    if lb_ub[0] > lb_ub[1]:

                        raise ValueError(
                            f"'bounds' values for '{control_vector[i]}' is invalid lower bound ({lb_ub[0]}) greater than upper bound ({lb_ub[1]})"
                        )
        else:

            raise TypeError(f"'bounds' argument must be list-like object")

    else:

        bounds = []

        for name in control_vector:

            if name in name_parameters:

                ind = np.argwhere(name_parameters == name)

                bounds.append(
                    (setup._lb_parameters[ind].item(), setup._ub_parameters[ind].item())
                )

            elif name in name_states:

                ind = np.argwhere(name_states == name)

                bounds.append(
                    (setup._lb_states[ind].item(), setup._ub_states[ind].item())
                )

    return bounds


def _standardize_gauge(
    gauge: (list, tuple, set), setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT
) -> list:

    code = np.array(mesh.code.tobytes(order="F").decode().split())

    if gauge:

        if isinstance(gauge, (list, tuple, set)):

            gauge_imd = gauge.copy()

            gauge = list(gauge)

            for i, name in enumerate(gauge):

                if name in code:

                    ind = np.argwhere(code == name)

                    if np.all(input_data.qobs[ind, setup._optim_start_step :] < 0):

                        gauge_imd.remove(name)

                        warnings.warn(
                            f"gauge '{name}' has no available observed discharge. Removed from the optimization"
                        )

                else:

                    raise ValueError(
                        f"gauge code '{name}' does not belong to the list of 'Model' gauges code {code}"
                    )

            if len(gauge_imd) == 0:

                raise ValueError(
                    f"No available observed discharge for optimization at gauge {gauge}"
                )

            else:

                gauge = gauge_imd.copy()

        elif isinstance(gauge, str):

            if gauge == "all":

                gauge = list(code.copy())

                gauge_imd = gauge.copy()

                for ind, name in enumerate(code):

                    if np.all(input_data.qobs[ind, setup._optim_start_step :] < 0):

                        gauge_imd.remove(name)

                        warnings.warn(
                            f"gauge '{name}' has no available observed discharge. Removed from the optimization"
                        )

                if len(gauge_imd) == 0:

                    raise ValueError(
                        f"No available observed discharge for optimization at gauge {gauge}"
                    )

                else:

                    gauge = gauge_imd.copy()

            elif gauge == "downstream":

                ind = np.argmax(mesh.area)

                if np.all(input_data.qobs[ind, setup._optim_start_step :] < 0):

                    raise ValueError(
                        f"No available observed discharge for optimization at gauge {gauge}"
                    )

                else:

                    gauge = [code[ind]]

            elif gauge in code:

                ind = np.argwhere(code == gauge)

                if np.all(input_data.qobs[ind, setup._optim_start_step :] < 0):

                    raise ValueError(
                        f"No available observed discharge for optimization at gauge {gauge}"
                    )

                else:

                    gauge = [gauge]

            else:

                raise ValueError(
                    f"str 'gauge' argument must be either one alias among ['all', 'downstream'] or a gauge code {code}"
                )

        else:
            raise TypeError(f"'gauge' argument must be list-like object or str")

    else:

        ind = np.argmax(mesh.area)

        if np.all(input_data.qobs[ind, setup._optim_start_step :] < 0):

            raise ValueError(
                f"No available observed discharge for optimization at gauge {code[ind]}"
            )

        else:

            gauge = [code[ind]]

    return gauge


def _standardize_wgauge(
    wgauge: (list, tuple, set), gauge: (list, tuple, set), mesh: MeshDT
) -> list:

    code = np.array(mesh.code.tobytes(order="F").decode().split())

    if wgauge:

        if isinstance(wgauge, (list, tuple, set)):

            imd_wgauge = list(wgauge)

            if len(imd_wgauge) != len(gauge):

                raise ValueError(
                    f"Inconsistent size between 'gauge' ({len(gauge)}) and 'wgauge' ({len(imd_wgauge)})"
                )

            else:

                wgauge = np.zeros(shape=len(code), dtype=np.float32)

                ind = np.in1d(code, gauge)

                wgauge[ind] = imd_wgauge

        elif isinstance(wgauge, (int, float)):

            imd_wgauge = [wgauge]

            if len(imd_wgauge) != len(gauge):

                raise ValueError(
                    f"Inconsistent size between 'gauge' ({len(gauge)}) and 'wgauge' ({len(imd_wgauge)})"
                )

            else:

                wgauge = np.zeros(shape=len(code), dtype=np.float32)

                ind = np.in1d(code, gauge)

                wgauge[ind] = imd_wgauge

        elif isinstance(wgauge, str):

            if wgauge == "mean":

                wgauge = np.zeros(shape=len(code), dtype=np.float32)

                wgauge = np.where(np.in1d(code, gauge), 1 / len(gauge), wgauge)

            elif wgauge == "area":

                wgauge = np.zeros(shape=len(code), dtype=np.float32)

                ind = np.in1d(code, gauge)

                value = mesh.area[ind] / sum(mesh.area[ind])

                wgauge[ind] = value

            elif wgauge == "area_minv":

                wgauge = np.zeros(shape=len(code), dtype=np.float32)

                ind = np.in1d(code, gauge)

                value = (1 / mesh.area[ind]) / sum(1 / mesh.area[ind])

                wgauge[ind] = value

            else:

                raise ValueError(
                    f"str 'wgauge' argument must be either one alias among ['mean', 'area', 'area_minv']"
                )

        else:

            raise TypeError(
                f"'wgauge' argument must be list-like object, int, float or str"
            )

    else:

        #% Default is mean
        wgauge = np.zeros(shape=len(code), dtype=np.float32)

        wgauge = np.where(np.in1d(code, gauge), 1 / len(gauge), wgauge)

    return wgauge


def _standardize_ost(ost: str, setup: SetupDT) -> pd.Timestamp:

    st = pd.Timestamp(setup.start_time.decode().strip())
    et = pd.Timestamp(setup.end_time.decode().strip())

    if ost:

        try:
            ost = pd.Timestamp(ost)

            if (ost - st).total_seconds() < 0 or (et - ost).total_seconds() < 0:

                raise ValueError(
                    f"'ost' ({ost}) argument must be between start time ({st}) and end time ({et})"
                )

        except:

            raise ValueError(f"'ost' ({ost}) argument is an invalid date")

    else:

        ost = st

    return ost


def _check_unknown_options(unknown_options: dict):

    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        warnings.warn("Unknown algorithm options: '%s'" % msg)


def _standardize_optimize_args(
    algorithm,
    control_vector,
    jobs_fun,
    mapping,
    bounds,
    gauge,
    wgauge,
    ost,
    setup,
    mesh,
    input_data,
):

    algorithm = _standardize_algorithm(algorithm)

    control_vector = _standardize_control_vector(control_vector, algorithm)

    jobs_fun = _standardize_jobs_fun(jobs_fun, algorithm)

    mapping = _standardize_mapping(mapping, algorithm, setup)

    bounds = _standardize_bounds(bounds, control_vector, setup)

    gauge = _standardize_gauge(gauge, setup, mesh, input_data)

    wgauge = _standardize_wgauge(wgauge, gauge, mesh)

    ost = _standardize_ost(ost, setup)

    return algorithm, control_vector, jobs_fun, mapping, bounds, wgauge, ost


def _standardize_optimize_options(options) -> dict:

    if options is None:

        options = {}

    return options
