from __future__ import annotations

from smash.solver._mwd_setup import SetupDT, Optimize_SetupDT
from smash.solver._mw_forward import forward
from smash.solver._mw_adjoint_test import scalar_product_test
from smash.solver._mw_optimize import (
    optimize_sbs,
    optimize_lbfgsb,
    hyper_optimize_lbfgsb,
)

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

STRUCTURE_PARAMETERS = {
    "gr-a": ["cp", "cft", "exc", "lr"],
    "gr-b": ["ci", "cp", "cft", "cst", "exc", "lr"],
    "vic-a": ["b", "cusl1", "cusl2", "clsl", "ks", "ds", "dsm", "ws", "lr"],
}

STRUCTURE_STATES = {
    "gr-a": ["hp", "hft", "hlr"],
    "gr-b": ["hi", "hp", "hft", "hst", "hlr"],
    "vic-a": ["husl1", "husl2", "hlsl"],
}


def _optimize_sbs(
    instance: Model,
    control_vector: np.ndarray,
    mapping: str,
    jobs_fun: str,
    bounds: np.ndarray,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    maxiter: int = 100,
    **unknown_options,
):

    _check_unknown_options(unknown_options)

    #% Reset default values
    instance.setup._optimize = Optimize_SetupDT(instance.setup, instance.mesh.ng, mapping)

    instance.setup._optimize.algorithm = "sbs"

    for i, name in enumerate(control_vector):

        if name in instance.setup._parameters_name:

            ind = np.argwhere(instance.setup._parameters_name == name)

            instance.setup._optimize.optim_parameters[ind] = 1

            instance.setup._optimize.lb_parameters[ind] = bounds[i, 0]
            instance.setup._optimize.ub_parameters[ind] = bounds[i, 1]

        #% Already check, must be states if not parameters
        else:

            ind = np.argwhere(instance.setup._states_name == name)

            instance.setup._optimize.optim_states[ind] = 1

            instance.setup._optimize.lb_states[ind] = bounds[i, 0]
            instance.setup._optimize.ub_states[ind] = bounds[i, 1]

    instance.setup._optimize.jobs_fun = jobs_fun

    instance.setup._optimize.wgauge = wgauge

    st = pd.Timestamp(instance.setup.start_time)

    instance.setup._optimize.optimize_start_step = (
        ost - st
    ).total_seconds() / instance.setup.dt + 1

    instance.setup._optimize.maxiter = maxiter

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
    control_vector: np.ndarray,
    mapping: str,
    jobs_fun: str,
    bounds: np.ndarray,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    maxiter: int = 100,
    jreg_fun: str = "prior",
    wjreg: float = 0.0,
    adjoint_test: bool = False,
    **unknown_options,
):

    _check_unknown_options(unknown_options)

    #% Reset default values
    instance.setup._optimize = Optimize_SetupDT(instance.setup, instance.mesh.ng, mapping)

    instance.setup._optimize.algorithm = "l-bfgs-b"

    for i, name in enumerate(control_vector):

        if name in instance.setup._parameters_name:

            ind = np.argwhere(instance.setup._parameters_name == name)

            instance.setup._optimize.optim_parameters[ind] = 1

            instance.setup._optimize.lb_parameters[ind] = bounds[i, 0]
            instance.setup._optimize.ub_parameters[ind] = bounds[i, 1]

        #% Already check, must be states if not parameters
        else:

            ind = np.argwhere(instance.setup._states_name == name)

            instance.setup._optimize.optim_states[ind] = 1

            instance.setup._optimize.lb_states[ind] = bounds[i, 0]
            instance.setup._optimize.ub_states[ind] = bounds[i, 1]

    instance.setup._optimize.jobs_fun = jobs_fun

    instance.setup._optimize.wgauge = wgauge

    st = pd.Timestamp(instance.setup.start_time)

    instance.setup._optimize.optimize_start_step = (
        ost - st
    ).total_seconds() / instance.setup.dt + 1

    instance.setup._optimize.maxiter = maxiter
    instance.setup._optimize.jreg_fun = jreg_fun
    instance.setup._optimize.wjreg = wjreg

    if instance.setup._optimize.mapping.startswith("hyper"):

        #% Add Adjoint test for hyper

        _optimize_message(instance, control_vector, mapping)

        hyper_optimize_lbfgsb(
            instance.setup,
            instance.mesh,
            instance.input_data,
            instance.parameters,
            instance.states,
            instance.output,
        )

    else:

        if adjoint_test:

            scalar_product_test(
                instance.setup,
                instance.mesh,
                instance.input_data,
                instance.parameters,
                instance.states,
                instance.output,
            )

        _optimize_message(instance, control_vector, mapping)

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
    control_vector: np.ndarray,
    mapping: str,
    jobs_fun: str,
    bounds: np.ndarray,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    maxiter: int | None = None,
    maxfev: int | None = None,
    disp: bool = False,
    return_all: bool = False,
    initial_simplex: np.ndarray | None = None,
    xatol: float = 0.0001,
    fatol: float = 0.0001,
    adaptive: bool = True,
    **unknown_options,
):
    global callback_args

    _check_unknown_options(unknown_options)

    #% Reset default values
    instance.setup._optimize = Optimize_SetupDT(instance.setup, instance.mesh.ng, mapping)

    instance.setup._optimize.algorithm = "nelder-mead"

    instance.setup._optimize.jobs_fun = jobs_fun

    instance.setup._optimize.wgauge = wgauge

    st = pd.Timestamp(instance.setup.start_time)

    instance.setup._optimize.optimize_start_step = (
        ost - st
    ).total_seconds() / instance.setup.dt + 1

    _optimize_message(instance, control_vector, mapping)

    callback_args = {"iterate": 0, "nfg": 0, "J": 0}

    if mapping == "uniform":

        parameters_bgd = instance.parameters.copy()
        states_bgd = instance.states.copy()

        x = _parameters_states_to_x(instance, control_vector)

        _problem(x, instance, control_vector, parameters_bgd, states_bgd)

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

        min_descriptor = np.empty(shape=instance.setup._nd, dtype=np.float32)
        max_descriptor = np.empty(shape=instance.setup._nd, dtype=np.float32)

        _normalize_descriptor(instance, min_descriptor, max_descriptor)

        x = _hyper_parameters_states_to_x(instance, control_vector, bounds)

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

        _denormalize_descriptor(instance, min_descriptor, max_descriptor)

    _callback(res.x)

    if res.success:

        print(f"{' ' * 4}CONVERGENCE: (XATOL, FATOL) < ({xatol}, {fatol})")

    else:

        print(f"{' ' * 4}STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT")


def _optimize_message(instance: Model, control_vector: np.ndarray, mapping: str):

    sp4 = " " * 4

    algorithm = instance.setup._optimize.algorithm
    jobs_fun = instance.setup._optimize.jobs_fun
    jreg_fun = instance.setup._optimize.jreg_fun
    wjreg = instance.setup._optimize.wjreg
    parameters = [el for el in control_vector if el in instance.setup._parameters_name]
    states = [el for el in control_vector if el in instance.setup._states_name]
    code = [
        el
        for ind, el in enumerate(instance.mesh.code)
        if instance.setup._optimize.wgauge[ind] > 0
    ]
    gauge = ["{:.6f}".format(el) for el in instance.setup._optimize.wgauge if el > 0]

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


def _parameters_states_to_x(instance: Model, control_vector: np.ndarray) -> np.ndarray:

    ac_ind = np.unravel_index(
        np.argmax(instance.mesh.active_cell, axis=None), instance.mesh.active_cell.shape
    )

    x = np.zeros(shape=control_vector.size, dtype=np.float32)

    for ind, name in enumerate(control_vector):

        if name in instance.setup._parameters_name:

            x[ind] = getattr(instance.parameters, name)[ac_ind]

        else:

            x[ind] = getattr(instance.states, name)[ac_ind]

    return x


def _hyper_parameters_states_to_x(
    instance: Model,
    control_vector: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:

    ac_ind = np.unravel_index(
        np.argmax(instance.mesh.active_cell, axis=None), instance.mesh.active_cell.shape
    )

    nd_step = 1 + instance.setup._nd

    x = np.zeros(shape=control_vector.size * nd_step, dtype=np.float32)

    for ind, name in enumerate(control_vector):

        lb, ub = bounds[ind, :]

        if name in instance.setup._parameters_name:

            y = getattr(instance.parameters, name)[ac_ind]

        else:

            y = getattr(instance.states, name)[ac_ind]

        x[ind * nd_step] = np.log((y - lb) / (ub - y))

    return x


def _x_to_parameters_states(x: np.ndarray, instance: Model, control_vector: np.ndarray):

    for ind, name in enumerate(control_vector):

        if name in instance.setup._parameters_name:

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
    x: np.ndarray, instance: Model, control_vector: np.ndarray, bounds: np.ndarray
):

    nd_step = 1 + instance.setup._nd

    for ind, name in enumerate(control_vector):

        value = x[ind * nd_step] * np.ones(
            shape=(instance.mesh.nrow, instance.mesh.ncol), dtype=np.float32
        )

        for i in range(instance.setup._nd):

            value += x[ind * nd_step + (i + 1)] * instance.input_data.descriptor[..., i]

        lb, ub = bounds[ind, :]

        y = (ub - lb) * (1.0 / (1.0 + np.exp(-value))) + lb

        if name in instance.setup._parameters_name:

            setattr(instance.parameters, name, y)

        else:

            setattr(instance.states, name, y)


def _normalize_descriptor(
    instance: Model, min_descriptor: np.ndarray, max_descriptor: np.ndarray
):

    for i in range(instance.setup._nd):
        min_descriptor[i] = np.amin(instance.input_data.descriptor[..., i])
        max_descriptor[i] = np.amax(instance.input_data.descriptor[..., i])

        instance.input_data.descriptor[..., i] = (
            instance.input_data.descriptor[..., i] - min_descriptor[i]
        ) / (max_descriptor[i] - min_descriptor[i])


def _denormalize_descriptor(
    instance: Model, min_descriptor: np.ndarray, max_descriptor: np.ndarray
):

    for i in range(instance.setup._nd):
        instance.input_data.descriptor[..., i] = (
            instance.input_data.descriptor[..., i]
            * (max_descriptor[i] - min_descriptor[i])
            + min_descriptor[i]
        )


def _problem(
    x: np.ndarray,
    instance: Model,
    control_vector: np.ndarray,
    parameters_bgd: ParametersDT,
    states_bgd: StatesDT,
):

    global callback_args

    _x_to_parameters_states(x, instance, control_vector)

    cost = np.float32(0)

    forward(
        instance.setup,
        instance.mesh,
        instance.input_data,
        instance.parameters,
        parameters_bgd,
        instance.states,
        states_bgd,
        instance.output,
        cost,
    )

    callback_args["nfg"] += 1
    callback_args["J"] = instance.output.cost

    return instance.output.cost


def _hyper_problem(
    x: np.ndarray,
    instance: Model,
    control_vector: np.ndarray,
    parameters_bgd: ParametersDT,
    states_bgd: StatesDT,
    bounds: np.ndarray,
):

    global callback_args

    _x_to_hyper_parameters_states(x, instance, control_vector, bounds)

    cost = np.float32(0)

    forward(
        instance.setup,
        instance.mesh,
        instance.input_data,
        instance.parameters,
        parameters_bgd,
        instance.states,
        states_bgd,
        instance.output,
        cost,
    )

    callback_args["nfg"] += 1
    callback_args["J"] = instance.output.cost

    return instance.output.cost


def _callback(x: np.ndarray, *args):

    global callback_args

    sp4 = " " * 4

    ret = []

    ret.append(f"{sp4}At iterate")
    ret.append("{:3}".format(callback_args["iterate"]))
    ret.append("nfg =" + "{:5}".format(callback_args["nfg"]))
    ret.append("J =" + "{:10.6f}".format(callback_args["J"]))

    callback_args["iterate"] += 1

    print(sp4.join(ret))


def _standardize_mapping(mapping: str | None, setup: SetupDT) -> str:

    #% Default values
    if mapping is None:

        mapping = "uniform"

    else:

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


def _standardize_jobs_fun(jobs_fun: str | None, algorithm: str) -> str:

    if jobs_fun is None:

        jobs_fun = "nse"

    else:

        if isinstance(jobs_fun, str):

            jobs_fun = jobs_fun.lower()

        else:

            raise TypeError(f"jobs_fun argument must be str")

        if jobs_fun not in JOBS_FUN:

            raise ValueError(
                f"Unknown objective function '{jobs_fun}'. Choices: {JOBS_FUN}"
            )

        elif jobs_fun in ["kge"] and algorithm == "l-bfgs-b":

            raise ValueError(
                f"'{jobs_fun}' objective function can not be use with '{algorithm}' algorithm (non convex function)"
            )

    return jobs_fun


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
    gauge: str | list | tuple | set | None,
    setup: SetupDT,
    mesh: MeshDT,
    input_data: Input_DataDT,
) -> np.ndarray:

    #% Default values
    if gauge is None:

        ind = np.argmax(mesh.area)

        if np.all(input_data.qobs[ind, setup._optimize.optimize_start_step :] < 0):

            raise ValueError(
                f"No available observed discharge for optimization at gauge {mesh.code[ind]}"
            )

        else:

            gauge = np.array(mesh.code[ind], ndmin=1)

    else:

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

                if np.all(
                    input_data.qobs[ind, setup._optimize.optimize_start_step :] < 0
                ):

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
    wgauge: str | list | tuple | set | None, gauge: np.ndarray, mesh: MeshDT
) -> np.ndarray:

    weight_arr = np.zeros(shape=mesh.code.size, dtype=np.float32)
    ind = np.in1d(mesh.code, gauge)

    #% Default values
    if wgauge is None:

        weight_arr[ind] = 1 / gauge.size

    else:

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


def _check_unknown_options(unknown_options: dict):

    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        warnings.warn("Unknown algorithm options: '%s'" % msg)


def _standardize_optimize_args(
    mapping: str | None,
    algorithm: str | None,
    control_vector: str | list | tuple | set | None,
    jobs_fun: str | None,
    bounds: list | tuple | set | None,
    gauge: str | list | tuple | set | None,
    wgauge: str | list | tuple | set | None,
    ost: str | pd.Timestamp | None,
    setup: SetupDT,
    mesh: MeshDT,
    input_data: Input_DataDT,
):

    mapping = _standardize_mapping(mapping, setup)

    algorithm = _standardize_algorithm(algorithm, mapping)

    control_vector = _standardize_control_vector(control_vector, setup)

    jobs_fun = _standardize_jobs_fun(jobs_fun, algorithm)

    bounds = _standardize_bounds(bounds, control_vector, setup)

    gauge = _standardize_gauge(gauge, setup, mesh, input_data)

    wgauge = _standardize_wgauge(wgauge, gauge, mesh)

    ost = _standardize_ost(ost, setup)

    return mapping, algorithm, control_vector, jobs_fun, bounds, wgauge, ost


def _standardize_optimize_options(options: dict | None) -> dict:

    if options is None:

        options = {}

    return options
