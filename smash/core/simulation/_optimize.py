from __future__ import annotations

from smash.solver._mw_forward import forward
from smash.solver._mw_adjoint_test import scalar_product_test
from smash.solver._mw_optimize import (
    optimize_sbs,
    optimize_lbfgsb,
    optimize_hyper_lbfgsb,
)

from smash.core._event_segmentation import _mask_event

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model
    from smash.solver._mwd_parameters import ParametersDT
    from smash.solver._mwd_states import StatesDT

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
    """
    Optimize the Model using the step-by-step (sbs) algorithm.

    Options
    -------
    maxiter : int, default 100
        Maximum allowed number of iterations.
    """

    _check_unknown_options(unknown_options)

    # % Fortran verbose
    instance.setup._optimize.verbose = verbose

    # % send mask_event to Fortran in case of event signatures based optimization
    if any([fn[0] == "E" for fn in jobs_fun]):
        instance.setup._optimize.mask_event = _mask_event(instance, **event_seg)

    instance.setup._optimize.algorithm = "sbs"

    for i, name in enumerate(control_vector):
        if name in instance.setup._parameters_name:
            ind = np.argwhere(instance.setup._parameters_name == name)

            instance.setup._optimize.optim_parameters[ind] = 1

            instance.setup._optimize.lb_parameters[ind] = bounds[i, 0]
            instance.setup._optimize.ub_parameters[ind] = bounds[i, 1]

        # % Already check, must be states if not parameters
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
    jreg_fun: np.ndarray | None = None,
    wjreg: float = 0.01,
    wjreg_fun: np.ndarray | None = None,
    auto_wjreg: str | None = None,
    nb_wjreg_lcurve: int = 6,
    return_lcurve: bool = False,
    adjoint_test: bool = False,
    **unknown_options,
):
    """
    Optimize the Model using the Limited-memory Broyden-Fletcher-Goldfarb-Shanno Bounded (L-BFGS-B) algorithm.

    Options
    -------
    maxiter : int, default 100
        Maximum allowed number of iterations.

    jreg_fun : str, sequence or None, default None
        Type(s) of regularization function(s) to be minimized when using distributed mapping. Should be one or a sequence of any of

        - 'prior'
        - 'smoothing'

        .. note::
            If not given, the Model will be optimized without any regularization terms.

        .. hint::
            See a detailed explanation on the regularization function in :ref:`Math / Num Documentation <math_num_documentation.signal_analysis.cost_functions>` section.

    wjreg: float, default 0.01
        Global regularization weight. Only used if **jreg_fun** is set.

    wjreg_fun : sequence or None, default None
        Regularization function(s) weights in case of multi-regularization (i.e. a sequence of regularization functions to minimize). Only used if **jreg_fun** is set.

        .. note::
            If not given and **jreg_fun** is set with multi-regularization, the weights are set to 1.

    auto_wjreg : str or None, default None
        Type of method to automatically compute the regularization function weight(s) **wjreg_fun**. Only used if **jreg_fun** is set. Should be one of

        - 'fast' (it consists of 2 optimization cycles)
        - 'lcurve' (it consists of **nb_wjreg_lcurve** optimization cycles)

        .. note::
            If not given, the regularization weight(s) are computed using the values set in **wjreg_fun**.

    nb_wjreg_lcurve : int, default 6
        Number of optimization cycles during the process of automatically computing the regularization function weight(s) using the L-curve approach. Only used if **jreg_fun** is set and **auto_wjreg** = 'lcurve'. The minimum value required is 6.

    return_lcurve : bool, default False
        If True, also return the L-curve results in the form of a dictionary. Only used if **jreg_fun** is set and **auto_wjreg** = 'lcurve'.

    adjoint_test : bool, default False
        If True, perform an adjoint test (i.e. a scalar product test). Only used if **mapping** = 'distributed'.
    """

    _check_unknown_options(unknown_options)

    # % Fortran verbose
    instance.setup._optimize.verbose = verbose

    # % send mask_event to Fortran in case of event signatures based optimization
    if any([fn[0] == "E" for fn in jobs_fun]):
        instance.setup._optimize.mask_event = _mask_event(instance, **event_seg)

    instance.setup._optimize.algorithm = "l-bfgs-b"

    for i, name in enumerate(control_vector):
        if name in instance.setup._parameters_name:
            ind = np.argwhere(instance.setup._parameters_name == name)

            instance.setup._optimize.optim_parameters[ind] = 1

            instance.setup._optimize.lb_parameters[ind] = bounds[i, 0]
            instance.setup._optimize.ub_parameters[ind] = bounds[i, 1]

        # % Already check, must be states if not parameters
        else:
            ind = np.argwhere(instance.setup._states_name == name)

            instance.setup._optimize.optim_states[ind] = 1

            instance.setup._optimize.lb_states[ind] = bounds[i, 0]
            instance.setup._optimize.ub_states[ind] = bounds[i, 1]

    instance.setup._optimize.jobs_fun = jobs_fun

    instance.setup._optimize.wjobs_fun = wjobs_fun

    if jreg_fun is not None:
        instance.setup._optimize.jreg_fun = jreg_fun

        instance.setup._optimize.wjreg = wjreg
    else:
        instance.setup._optimize.wjreg = 0

    if wjreg_fun is not None:
        instance.setup._optimize.wjreg_fun = wjreg_fun

    # %TODO: Add "distance_correlation" once clearly verified
    # ~ if not (type(reg_descriptors)==type(None)):
    # ~ instance.setup._optimize.reg_descriptors_for_params = reg_descriptors["params"]
    # ~ instance.setup._optimize.reg_descriptors_for_states = reg_descriptors["states"]

    instance.setup._optimize.wgauge = wgauge

    st = pd.Timestamp(instance.setup.start_time)

    instance.setup._optimize.optimize_start_step = (
        ost - st
    ).total_seconds() / instance.setup.dt + 1

    instance.setup._optimize.maxiter = maxiter

    if instance.setup._optimize.mapping.startswith("hyper"):
        # % TODO: Add Adjoint test for hyper

        if verbose:
            _optimize_message(instance, control_vector, mapping)

        optimize_hyper_lbfgsb(
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

        if auto_wjreg == "fast":
            parameters_bgd = instance.parameters.copy()
            states_bgd = instance.states.copy()

            instance.setup._optimize.wjreg = 0.0

            if verbose:
                _optimize_message(instance, control_vector, mapping)
                print(
                    f"{' '*4}CYCLE 1: wJreg = {'{:.6f}'.format(instance.setup._optimize.wjreg)}\n"
                )

            optimize_lbfgsb(
                instance.setup,
                instance.mesh,
                instance.input_data,
                instance.parameters,
                instance.states,
                instance.output,
            )

            # compute the best wjreg with the fast method
            instance.setup._optimize.wjreg = (
                instance.output._cost_jobs_initial - instance.output.cost_jobs
            ) / (instance.output.cost_jreg)

            instance.parameters = parameters_bgd.copy()
            instance.states = states_bgd.copy()

            if verbose:
                print(
                    f"{' '*4}FINAL CYCLE: wJreg = {'{:.6f}'.format(instance.setup._optimize.wjreg)}\n"
                )

            optimize_lbfgsb(
                instance.setup,
                instance.mesh,
                instance.input_data,
                instance.parameters,
                instance.states,
                instance.output,
            )

        elif auto_wjreg == "lcurve":
            # % first optimization with wjreg = 0.
            parameters_bgd = instance.parameters.copy()
            states_bgd = instance.states.copy()

            n_cycle = 1

            instance.setup._optimize.wjreg = 0.0

            if verbose:
                _optimize_message(instance, control_vector, mapping)
                print(
                    f"{' '*4}CYCLE {n_cycle}: wJreg = {'{:.6f}'.format(instance.setup._optimize.wjreg)}\n"
                )

            optimize_lbfgsb(
                instance.setup,
                instance.mesh,
                instance.input_data,
                instance.parameters,
                instance.states,
                instance.output,
            )
            
            n_cycle += 1
            
            # % bounds initialisation for jobs and jreg
            jobs_min = instance.output.cost_jobs
            jobs_max = instance.output._cost_jobs_initial
            jreg_min = 0.0 #minimum jreg value, instance.output._cost_jreg_initial > 0 if hard-smoothing
            jreg_max = instance.output.cost_jreg

            if (jobs_min / jobs_max) < 0.95 and (jreg_max - jreg_min) > 0.0:
                # Computation of the best wjreg using the "fast" method
                wjreg_opt = (jobs_max - jobs_min) / (jreg_max)
                # Computation of the range of wjreg centered on wjreg_opt (4 points minimum)
                wjreg_range = _compute_wjreg_range(wjreg_opt, nb_wjreg_lcurve)

            else:
                wjreg_opt = 0.0
                wjreg_range = np.empty(shape=0)

            # array initialisation
            cost_arr = np.zeros(shape=wjreg_range.size + 1, dtype=np.float32)
            cost_arr[0] = instance.output.cost

            cost_jobs_arr = np.zeros(shape=wjreg_range.size + 1, dtype=np.float32)
            cost_jobs_arr[0] = instance.output.cost_jobs

            cost_jreg_arr = np.zeros(shape=wjreg_range.size + 1, dtype=np.float32)
            cost_jreg_arr[0] = instance.output.cost_jreg

            wjreg_arr = np.zeros(shape=wjreg_range.size + 1, dtype=np.float32)
            wjreg_arr[0] = instance.setup._optimize.wjreg

            # Doing the lcurve with wjreg_range for optimization
            for i, wj in enumerate(wjreg_range):
                instance.setup._optimize.wjreg = wj

                instance.parameters = parameters_bgd.copy()
                instance.states = states_bgd.copy()

                if verbose:
                    print(
                        f"{' '*4}CYCLE {n_cycle}: wJreg = {'{:.6f}'.format(instance.setup._optimize.wjreg)}\n"
                    )

                optimize_lbfgsb(
                    instance.setup,
                    instance.mesh,
                    instance.input_data,
                    instance.parameters,
                    instance.states,
                    instance.output,
                )

                n_cycle += 1

                cost_arr[i + 1] = instance.output.cost
                cost_jobs_arr[i + 1] = instance.output.cost_jobs
                cost_jreg_arr[i + 1] = instance.output.cost_jreg
                wjreg_arr[i + 1] = instance.setup._optimize.wjreg

                # % break if jobs does not minimize
                # Bug, todo: if we break, we should reshape the array
                # ~ if (instance.output.cost_jobs - jobs_min) / (
                    # ~ jobs_max - jobs_min
                # ~ ) >= 0.8:
                    # ~ break

            # % bounds update for jobs and jreg
            jobs_min = np.min(cost_jobs_arr)
            jobs_max = np.max(cost_jobs_arr)
            jreg_max = np.max(cost_jreg_arr)
            jreg_min = np.min(cost_jreg_arr)
            
            # % select the best wjreg based on the transformed lcurve and using our own method decribed in ...
            distance, wjreg_lcurve_opt = _compute_best_lcurve_weight(
                cost_jobs_arr,
                cost_jreg_arr,
                wjreg_arr,
                jobs_min,
                jobs_max,
                jreg_min,
                jreg_max,
            )

            # % save the lcurve
            lcurve = {
                "cost_jobs_initial": jobs_max,
                "cost_jreg_initial": jreg_min,
                "wjreg_lcurve_opt": wjreg_lcurve_opt,
                "wjreg_fast": wjreg_opt,
                "wjreg": wjreg_arr,
                "distance": distance,
                "cost": cost_arr,
                "cost_jobs": cost_jobs_arr,
                "cost_jreg": cost_jreg_arr,
            }

            # % last optim with best wjreg
            if wjreg_lcurve_opt is not None:
                instance.parameters = parameters_bgd.copy()
                instance.states = states_bgd.copy()

                instance.setup._optimize.wjreg = wjreg_lcurve_opt

                if verbose:
                    print(
                        f"{' '*4}FINAL CYCLE: wJreg = {'{:.6f}'.format(instance.setup._optimize.wjreg)}\n"
                    )

                optimize_lbfgsb(
                    instance.setup,
                    instance.mesh,
                    instance.input_data,
                    instance.parameters,
                    instance.states,
                    instance.output,
                )

            else:
                instance.parameters = parameters_bgd.copy()
                instance.states = states_bgd.copy()
                cost = np.float32(0)

                # % run the model as it is
                forward(
                    instance.setup,
                    instance.mesh,
                    instance.input_data,
                    instance.parameters,
                    instance.parameters.copy(),
                    instance.states,
                    instance.states.copy(),
                    instance.output,
                    cost,
                )

            if return_lcurve:
                return lcurve

        else:
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
    """
    Optimize the Model using the Nelder-Mead algorithm.

    Options
    -------
    maxiter, maxfev : int or None, default None
        Maximum allowed number of iterations and function evaluations.

        .. note::
            If not given, default to ``N*200``, where ``N`` is the number of
            variables, if neither maxiter or maxfev is set. If both
            maxiter and maxfev are set, minimization will stop at the
            first reached.

    disp : bool, default False
        If True, display convergence messages.

    return_all : bool, default False
        Set to True to return a list of the best solution at each of the
        iterations.

        .. warning::
            Not working at the moment.

    initial_simplex : np.ndarray of shape (N + 1, N) or None, default None
        Initial simplex.

        .. note::
            If given, overrides prior control.
            ``initial_simplex[j,:]`` should contain the coordinates of
            the jth vertex of the ``N+1`` vertices in the simplex, where
            ``N`` is the dimension.

    xatol : float, default 0.0001
        Absolute error in xopt between iterations that is acceptable for
        convergence.

    fatol : float, default 0.0001
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.

    adaptive : bool, default True
        Adapt algorithm parameters to dimensionality of problem. Useful for
        high-dimensional minimization.
    """

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
    wjreg_fun = instance.setup._optimize.wjreg_fun
    wjreg = instance.setup._optimize.wjreg
    parameters = [el for el in control_vector if el in instance.setup._parameters_name]
    states = [el for el in control_vector if el in instance.setup._states_name]
    code = [
        el
        for ind, el in enumerate(instance.mesh.code)
        if instance.setup._optimize.wgauge[ind] != 0
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
    ret.append(f"wJobs function: [ {' '.join(wjobs_fun.astype('U'))} ]")

    if algorithm == "l-bfgs-b":
        ret.append(f"Jreg function: [ {' '.join(jreg_fun)} ]")
        ret.append(f"wJreg function: [ {' '.join(wjreg_fun.astype('U'))} ]")
        ret.append(f"wJreg: {wjreg}")

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
        raise KeyError("Unknown algorithm options: '%s'" % msg)


def _compute_wjreg_range(wjreg_opt: float, nb_wjreg_lcurve: int):
    # % Computation of the range of wjreg centered on wjreg_opt (4 points minimum)

    log_wjreg_opt = np.log10(wjreg_opt)
    nb_wjreg_lcurve_base = nb_wjreg_lcurve - 6

    # % 5 points minimum
    base = np.array(
        10 ** np.arange(log_wjreg_opt - 0.66, log_wjreg_opt + 0.67, 0.33),
        dtype=np.float32,
    )

    if nb_wjreg_lcurve_base > 0:
        min_wjreg = (
            log_wjreg_opt
            - 0.66
            - (nb_wjreg_lcurve_base - np.ceil(nb_wjreg_lcurve_base / 2.0))
        )
        max_wjreg = (
            log_wjreg_opt
            + 0.66
            + 1.0
            + (nb_wjreg_lcurve_base - np.floor(nb_wjreg_lcurve_base / 2.0))
        )
        lower = np.array(
            10 ** np.arange(min_wjreg, log_wjreg_opt - 0.66), dtype=np.float32
        )
        upper = np.array(
            10 ** np.arange(log_wjreg_opt + 0.66 + 1.0, max_wjreg), dtype=np.float32
        )

        wjreg_range = np.hstack((lower, base, upper))

    else:
        wjreg_range = base

    return wjreg_range


def _compute_best_lcurve_weight(
    cost_jobs_arr: np.ndarray,
    cost_jreg_arr: np.ndarray,
    wjreg_arr: np.ndarray,
    jobs_min: float,
    jobs_max: float,
    jreg_min: float,
    jreg_max: float,
):
    # % select the best wjreg based on the transformed lcurve and using our own method decribed in ...
    wjreg_lcurve_opt=None
    
    if (
        cost_jobs_arr.size > 2
        and (jreg_max - jreg_min) > 0.0
        and (jobs_max - jobs_min) > 0.0
    ):
        max_distance = 0.0
        distance = np.zeros(shape=cost_jobs_arr.size, dtype=np.float32)

        for i in range(cost_jobs_arr.size):
            # % skip point above y = x
            if ((cost_jreg_arr[i] - jreg_min) / (jreg_max - jreg_min)) < (
                (jobs_max - cost_jobs_arr[i]) / (jobs_max - jobs_min)
            ):
                if cost_jobs_arr[i] < jobs_max:
                    hypth = (
                        ((jobs_max - cost_jobs_arr[i]) / (jobs_max - jobs_min)) ** 2.0
                        + ((cost_jreg_arr[i] - jreg_min) / (jreg_max - jreg_min)) ** 2.0
                    ) ** 0.5
                    alpha = np.pi * 0.25 - np.arccos(
                        ((jobs_max - cost_jobs_arr[i]) / (jobs_max - jobs_min)) / hypth
                    )

                    distance[i] = hypth * np.sin(alpha)

                else:
                    distance[i] = 0.0

                if distance[i] >= max_distance:
                    max_distance = distance[i]
                    wjreg_lcurve_opt = wjreg_arr[i]
                
            else:
                distance[i] = np.nan
    else:
        distance = np.empty(shape=0)
        wjreg_lcurve_opt = None

    return distance, wjreg_lcurve_opt
