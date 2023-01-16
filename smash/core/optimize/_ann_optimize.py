from __future__ import annotations

from smash.core._event_segmentation import _mask_event

from smash.core.optimize._optimize import _normalize_descriptor, _denormalize_descriptor

from smash.core.net import Net

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

import numpy as np
import pandas as pd
import warnings


def _ann_optimize(
    instance: Model,
    control_vector: np.ndarray,
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
    event_seg: dict,
    bounds: np.ndarray,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    net: Net | None,
    validation: float | None,
    epochs: int,
    early_stopping: bool,
    verbose: bool,
):

    # send mask_event to Fortran in case of event signatures based optimization
    if any([fn[0] == "E" for fn in jobs_fun]):
        instance.setup._optimize.mask_event = _mask_event(instance, **event_seg)

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

    # initial parameters and states
    parameters_bgd = instance.parameters.copy()
    states_bgd = instance.states.copy()

    # preprocessing data
    min_descriptor = np.empty(shape=instance.setup._nd, dtype=np.float32)
    max_descriptor = np.empty(shape=instance.setup._nd, dtype=np.float32)

    active_mask = np.where(instance.mesh.active_cell == 1)
    inactive_mask = np.where(instance.mesh.active_cell == 0)

    _normalize_descriptor(instance, min_descriptor, max_descriptor)

    x_train = instance.input_data.descriptor[active_mask]
    x_inactive = instance.input_data.descriptor[inactive_mask]

    _denormalize_descriptor(instance, min_descriptor, max_descriptor)

    # set graph if not defined
    nd = instance.setup._nd
    nx = len(x_train)

    net = _set_graph(net, nx, nd, control_vector, bounds)

    if verbose:
        _training_message(instance, control_vector, nx, net)

    # train the network
    net._fit(
        x_train,
        instance,
        control_vector,
        active_mask,
        parameters_bgd,
        states_bgd,
        validation,
        epochs,
        early_stopping,
        verbose,
    )

    # predicted map at inactive cells
    y = net._predict(x_inactive)

    for i, name in enumerate(control_vector):

        if name in instance.setup._parameters_name:
            getattr(instance.parameters, name)[inactive_mask] = y[:, i]

        else:
            getattr(instance.states, name)[inactive_mask] = y[:, i]

    return net


def _set_graph(
    net: Net | None,
    ntrain: int,
    nd: int,
    control_vector: np.ndarray,
    bounds: np.ndarray,
):

    ncv = control_vector.size

    if net is None:  # auto-graph

        net = Net()

        n_neurons = round(np.sqrt(ntrain * nd) * 2 / 3)

        net.add(
            layer="dense",
            options={
                "input_shape": (nd,),
                "neurons": n_neurons,
                "kernel_initializer": "glorot_uniform",
            },
        )
        net.add(layer="activation", options={"name": "relu"})

        net.add(
            layer="dense",
            options={
                "neurons": round(n_neurons / 2),
                "kernel_initializer": "glorot_uniform",
            },
        )
        net.add(layer="activation", options={"name": "relu"})

        net.add(
            layer="dense",
            options={"neurons": ncv, "kernel_initializer": "glorot_uniform"},
        )
        net.add(layer="activation", options={"name": "sigmoid"})

        net.add(
            layer="scale",
            options={"bounds": bounds},
        )

        net.compile(optimizer="adam", learning_rate=0.001)

    elif not isinstance(net, Net):
        raise ValueError(f"Unknown network {net}")

    elif not net.layers:
        raise ValueError(f"The graph has not been set yet")

    else:
        #% check input shape
        ips = net.layers[0].input_shape

        if ips[0] != nd:

            raise ValueError(
                f"Inconsistent value between the number of input layer ({ips}) and the number of descriptors ({nd}): {ips[0]} != {nd}"
            )

        #% check output shape
        ios = net.layers[-1].output_shape()

        if ios[0] != ncv:

            raise ValueError(
                f"Inconsistent value between the number of output layer ({ios}) and the number of control vectors ({ncv}): {ios[0]} != {ncv}"
            )

        #% check bounds constraints
        if hasattr(net.layers[-1], "_scale_func"):

            net_bounds = net.layers[-1]._scale_func._bounds

            diff = np.not_equal(net_bounds, bounds)

            for i, name in enumerate(control_vector):

                if diff[i].any():

                    warnings.warn(
                        f"Inconsistent value(s) between scaling parameters ({net_bounds[i]}) and the bound constraints of control vector {name} ({bounds[i]}). Use get_bound_constraints method of Model instance to properly create scaling layer"
                    )

    return net


def _training_message(
    instance: Model,
    control_vector: np.ndarray,
    nx: int,
    net: Net,
):

    sp4 = " " * 4

    jobs_fun = instance.setup._optimize.jobs_fun
    wjobs_fun = instance.setup._optimize.wjobs_fun
    parameters = [el for el in control_vector if el in instance.setup._parameters_name]
    states = [el for el in control_vector if el in instance.setup._states_name]
    code = [
        el
        for ind, el in enumerate(instance.mesh.code)
        if instance.setup._optimize.wgauge[ind] > 0
    ]
    wgauge = np.array([el for el in instance.setup._optimize.wgauge if el > 0])

    mapping_eq = "k(x) = N(D1, ..., Dn)"
    len_parameters = len(parameters) * (1 + 2 * instance.setup._nd)
    len_states = len(states) * (1 + 2 * instance.setup._nd)

    ret = []

    ret.append(f"{sp4}Mapping: 'ANN' {mapping_eq}")

    ret.append(f"Optimizer: {net._optimizer}")
    ret.append(f"Learning rate: {net._learning_rate}")

    ret.append(f"Jobs function: [ {' '.join(jobs_fun)} ]")
    ret.append(f"wJobs: [ {' '.join(wjobs_fun.astype('U'))} ]")

    ret.append(f"Nx: {nx}")
    ret.append(f"Np: {len_parameters} [ {' '.join(parameters)} ]")
    ret.append(f"Ns: {len_states} [ {' '.join(states)} ]")
    ret.append(f"Ng: {len(code)} [ {' '.join(code)} ]")
    ret.append(f"wg: {len(wgauge)} [ {' '.join(wgauge.astype('U'))} ]")

    print(f"\n{sp4}".join(ret) + "\n")
