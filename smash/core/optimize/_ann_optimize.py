from __future__ import annotations

from smash.solver._mwd_setup import Optimize_SetupDT

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
    jreg_fun: str,
    wjreg: float,
    net: Net | None,
    validation: float | None,
    epochs: int,
    early_stopping: bool,
    verbose: bool,
):

    #% Reset default values
    instance.setup._optimize = Optimize_SetupDT(
        instance.setup, instance.mesh.ng, njf=len(jobs_fun)
    )

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

    instance.setup._optimize.jreg_fun = jreg_fun
    instance.setup._optimize.wjreg = wjreg

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

    net = _set_graph(net, nd, len(control_vector), bounds)
    _training_message(
        instance, control_vector, len(x_train), net.optimizer, net.learning_rate
    )

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


def _set_graph(net: Net | None, nd: int, ncv: int, bounds: np.ndarray):

    if net is None:  # set a default graph

        net = Net()

        net.add(layer="dense", options={"input_shape": (nd,), "neurons": 16})
        net.add(layer="activation", options={"name": "relu"})

        net.add(layer="dense", options={"neurons": 8})
        net.add(layer="activation", options={"name": "relu"})

        net.add(layer="dense", options={"neurons": ncv})
        net.add(layer="activation", options={"name": "sigmoid"})

        net.add(
            layer="scale",
            options={
                "name": "minmaxscale",
                "lower": bounds[:, 0],
                "upper": bounds[:, 1],
            },
        )

        net.compile()

    elif not isinstance(net, Net):
        raise ValueError(f"Unknown network {net}")

    elif len(net.layers) == 0:
        raise ValueError(f"Cannot train the network. The graph has not been set yet")

    else:
        pass

    return net


def _training_message(
    instance: Model, control_vector: np.ndarray, n_train: int, opt: str, lr: float
):

    sp4 = " " * 4

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

    mapping_eq = "k(x) = N(D1, ..., Dn)"
    len_parameters = len(parameters) * (1 + 2 * instance.setup._nd)
    len_states = len(states) * (1 + 2 * instance.setup._nd)

    ret = []

    ret.append("</> Optimize Model J")

    ret.append(f"Mapping: 'ANN' {mapping_eq}")

    ret.append(f"Training set size: {n_train}")
    ret.append(f"Optimizer: {opt}")
    ret.append(f"Learning rate: {lr}")

    ret.append(f"Jobs function: [ {' '.join(jobs_fun)} ]")
    ret.append(f"wJobs: [ {' '.join(wjobs_fun.astype('U'))} ]")
    ret.append(f"Jreg function: '{jreg_fun}'")
    ret.append(f"wJreg: {'{:.6f}'.format(wjreg)}")

    ret.append(f"Np: {len_parameters} [ {' '.join(parameters)} ]")
    ret.append(f"Ns: {len_states} [ {' '.join(states)} ]")
    ret.append(f"Ng: {len(code)} [ {' '.join(code)} ]")
    ret.append(f"wg: {len(wgauge)} [ {' '.join(wgauge.astype('U'))} ]")

    print(f"\n{sp4}".join(ret) + "\n")
