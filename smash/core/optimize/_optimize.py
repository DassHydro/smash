from __future__ import annotations

from smash.solver._mw_forward import forward
from smash.solver._mw_adjoint_test import scalar_product_test
from smash.solver._mw_optimize import (
    optimize_sbs,
    optimize_lbfgsb,
    hyper_optimize_lbfgsb,
)

from smash.core._event_segmentation import _mask_event

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model
    from smash.solver._mwd_parameters import ParametersDT
    from smash.solver._mwd_states import StatesDT

import warnings
import numpy as np
import scipy.optimize
import pandas as pd


def _optimize_sbs(
    instance: Model,
    control_vector: np.ndarray,
    mapping: str,
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
    event_seg: dict,
    bounds: np.ndarray,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    verbose: bool,
    maxiter: int = 100,
    **unknown_options,
):

    _check_unknown_options(unknown_options)

    #% Fortran verbose
    instance.setup._optimize.verbose = verbose

    # send mask_event to Fortran in case of event signatures based optimization
    if any([fn[0] == "E" for fn in jobs_fun]):
        instance.setup._optimize.mask_event = _mask_event(instance, **event_seg)

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

    instance.setup._optimize.wjobs_fun = wjobs_fun

    instance.setup._optimize.wgauge = wgauge

    st = pd.Timestamp(instance.setup.start_time)

    instance.setup._optimize.optimize_start_step = (
        ost - st
    ).total_seconds() / instance.setup.dt + 1

    instance.setup._optimize.maxiter = maxiter

    if verbose:
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
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
    event_seg: dict,
    bounds: np.ndarray,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    verbose: bool,
    maxiter: int = 100,
    jreg_fun: str = "prior",
    wjreg: float = 0.0,
    adjoint_test: bool = False,
    **unknown_options,
):

    _check_unknown_options(unknown_options)

    #% Fortran verbose
    instance.setup._optimize.verbose = verbose

    # send mask_event to Fortran in case of event signatures based optimization
    if any([fn[0] == "E" for fn in jobs_fun]):
        instance.setup._optimize.mask_event = _mask_event(instance, **event_seg)

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

    instance.setup._optimize.wjobs_fun = wjobs_fun

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

        if verbose:
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

        if verbose:
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
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
    event_seg: dict,
    bounds: np.ndarray,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    verbose: bool,
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

    # send mask_event to Fortran in case of event signatures based optimization
    if any([fn[0] == "E" for fn in jobs_fun]):
        instance.setup._optimize.mask_event = _mask_event(instance, **event_seg)

    instance.setup._optimize.algorithm = "nelder-mead"

    instance.setup._optimize.jobs_fun = jobs_fun

    instance.setup._optimize.wjobs_fun = wjobs_fun

    instance.setup._optimize.wgauge = wgauge

    st = pd.Timestamp(instance.setup.start_time)

    instance.setup._optimize.optimize_start_step = (
        ost - st
    ).total_seconds() / instance.setup.dt + 1

    if verbose:
        _optimize_message(instance, control_vector, mapping)

    callback_args = {"iterate": 0, "nfg": 0, "J": 0, "verbose": verbose}

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

    if verbose:

        if res.success:
            print(f"{' ' * 4}CONVERGENCE: (XATOL, FATOL) < ({xatol}, {fatol})")

        else:
            print(f"{' ' * 4}STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT")


def _optimize_message(instance: Model, control_vector: np.ndarray, mapping: str):

    sp4 = " " * 4

    algorithm = instance.setup._optimize.algorithm
    jobs_fun = instance.setup._optimize.jobs_fun
    wjobs_fun = instance.setup._optimize.wjobs_fun
    jreg_fun = instance.setup._optimize.jreg_fun
    wjreg = instance.setup._optimize.wjreg
    parameters = [el for el in control_vector if el in instance.setup._parameters_name]
    states = [el for el in control_vector if el in instance.setup._states_name]
    code = [
        el
        for ind, el in enumerate(instance.mesh.code)
        if instance.setup._optimize.wgauge[ind] > 0
    ]
    wgauge = np.array([el for el in instance.setup._optimize.wgauge if el > 0])

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

    ret.append(f"{sp4}Mapping: '{mapping}' {mapping_eq}")
    ret.append(f"Algorithm: '{algorithm}'")
    ret.append(f"Jobs function: [ {' '.join(jobs_fun)} ]")
    ret.append(f"wJobs: [ {' '.join(wjobs_fun.astype('U'))} ]")

    if algorithm == "l-bfgs-b":
        ret.append(f"Jreg function: '{jreg_fun}'")
        ret.append(f"wJreg: {'{:.6f}'.format(wjreg)}")

    ret.append(f"Nx: {nx}")
    ret.append(f"Np: {len_parameters} [ {' '.join(parameters)} ]")
    ret.append(f"Ns: {len_states} [ {' '.join(states)} ]")
    ret.append(f"Ng: {len(code)} [ {' '.join(code)} ]")
    ret.append(f"wg: {len(wgauge)} [ {' '.join(wgauge.astype('U'))} ]")

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

    if callback_args["verbose"]:

        sp4 = " " * 4

        ret = []

        ret.append(f"{sp4}At iterate")
        ret.append("{:3}".format(callback_args["iterate"]))
        ret.append("nfg =" + "{:5}".format(callback_args["nfg"]))
        ret.append("J =" + "{:10.6f}".format(callback_args["J"]))

        callback_args["iterate"] += 1

        print(sp4.join(ret))


def _check_unknown_options(unknown_options: dict):

    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        warnings.warn("Unknown algorithm options: '%s'" % msg)
