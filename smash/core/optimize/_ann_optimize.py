from __future__ import annotations

from smash.core._event_segmentation import _mask_event

from smash.core.net import Net

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

import numpy as np
import pandas as pd


def _ann_optimize(
    instance: Model,
    control_vector: np.ndarray,
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
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
        instance.setup._optimize.mask_event = _mask_event(instance)

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

    nd = instance.setup._nd

    # preprocessing data
    mask = np.where(instance.mesh.active_cell == 1)

    x_train = instance.input_data.descriptor.copy()
    x_train = x_train[mask]

    for i in range(nd):
        x_train[..., i] = (x_train[..., i] - np.nanmin(x_train[..., i])) / (
            np.nanmax(x_train[..., i]) - np.nanmin(x_train[..., i])
        )

    # set graph if not defined
    nx = len(x_train)
    net = _set_graph(net, nx, nd, len(control_vector), bounds)

    if verbose:
        _training_message(instance, control_vector, nx, net)

    # train the network
    net._fit(
        x_train,
        instance,
        control_vector,
        mask,
        parameters_bgd,
        states_bgd,
        validation,
        epochs,
        early_stopping,
        verbose,
    )

    return net


def _set_graph(net: Net | None, ntrain: int, nd: int, ncv: int, bounds: np.ndarray):

    if net is None:  # auto-graph

        net = Net()

        #% Net 1 =======================

        # n_hidden_layers = max(round(ntrain / (9 * (nd + ncv))), 1)

        # n_neurons = round(2 / 3 * nd + ncv)

        # for i in range(n_hidden_layers):

        #     if i == 0:

        #         net.add(
        #             layer="dense",
        #             options={
        #                 "input_shape": (nd,),
        #                 "neurons": n_neurons,
        #                 "kernel_initializer": "he_uniform",
        #             },
        #         )

        #     else:

        #         n_neurons_i = max(
        #             round((n_hidden_layers - i) / n_hidden_layers * n_neurons), ncv
        #         )
        #         net.add(
        #             layer="dense",
        #             options={
        #                 "neurons": n_neurons_i,
        #                 "kernel_initializer": "he_uniform",
        #             },
        #         )

        #     net.add(layer="activation", options={"name": "relu"})

        #% Net 2 =======================

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
        # net.add(layer="dropout", options={"drop_rate": .1})

        net.add(
            layer="dense",
            options={
                "neurons": round(n_neurons / 2),
                "kernel_initializer": "glorot_uniform",
            },
        )
        net.add(layer="activation", options={"name": "relu"})
        # net.add(layer="dropout", options={"drop_rate": .2})

        #% =============================

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
