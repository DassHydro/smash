from __future__ import annotations

from smash.core.simulation.optimize._standardize import (
    _standardize_multiple_optimize_args,
)

from smash.fcore._mw_optimize import (
    optimize as wrap_optimize,
    multiple_optimize as wrap_multiple_optimize,
)
from smash.fcore._mwd_options import OptionsDT
from smash.fcore._mwd_returns import ReturnsDT

import numpy as np
from copy import deepcopy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.factory.samples.samples import Samples


__all__ = ["MultipleOptimize", "optimize", "multiple_optimize"]


class MultipleOptimize(dict):
    """
    Represents multiple optimize computation result.

    Notes
    -----
    This class is essentially a subclass of dict with attribute accessors.

    TODO FC: Fill

    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return "\n".join(
                [
                    k.rjust(m) + ": " + repr(v)
                    for k, v in sorted(self.items())
                    if not k.startswith("_")
                ]
            )
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def optimize(
    model: Model,
    mapping: str = "uniform",
    optimizer: str | None = None,
    optimize_options: dict | None = None,
    cost_options: dict | None = None,
    common_options: dict | None = None,
):
    
    """
    Model assimilation using numerical optimization algorithms.

    TODO: Fill

    Returns
    -------
    ret_model : Model
        The optimized Model.
    """

    wmodel = model.copy()

    wmodel.optimize(mapping, optimizer, optimize_options, cost_options, common_options)

    return wmodel


def _optimize(
    model: Model,
    mapping: str,
    optimizer: str,
    optimize_options: dict,
    cost_options: dict,
    common_options: dict,
):
    if common_options["verbose"]:
        print("</> Optimize")

    wrap_options = OptionsDT(
        model.setup,
        model.mesh,
        cost_options["njoc"],
        cost_options["njrc"],
    )

    # % Map optimize_options dict to derived type
    for key, value in optimize_options.items():
        if hasattr(wrap_options.optimize, key):
            setattr(wrap_options.optimize, key, value)

        if key == "termination_crit":
            for key_termination_crit, value_termination_crit in value.items():
                if hasattr(wrap_options.optimize, key_termination_crit):
                    setattr(
                        wrap_options.optimize,
                        key_termination_crit,
                        value_termination_crit,
                    )

    # % Map cost_options dict to derived type
    for key, value in cost_options.items():
        if hasattr(wrap_options.cost, key):
            setattr(wrap_options.cost, key, value)

    # % Map common_options dict to derived type
    for key, value in common_options.items():
        if hasattr(wrap_options.comm, key):
            setattr(wrap_options.comm, key, value)

    # % TODO: Implement return options
    wrap_returns = ReturnsDT()

    if mapping == "ann":
        _ann_optimize(
            model,
            optimizer,
            optimize_options,
            common_options,
            wrap_options,
            wrap_returns,
        )

    else:
        wrap_optimize(
            model.setup,
            model.mesh,
            model._input_data,
            model._parameters,
            model._output,
            wrap_options,
            wrap_returns,
        )


def _ann_optimize(
    model: Model,
    optimizer: str,
    optimize_options: dict,
    common_options: dict,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
):
    # % preprocessing input descriptors and normalization
    active_mask = np.where(model.mesh.active_cell == 1)
    inactive_mask = np.where(model.mesh.active_cell == 0)

    l_desc = model._input_data.physio_data.l_descriptor
    u_desc = model._input_data.physio_data.u_descriptor

    desc = model._input_data.physio_data.descriptor.copy()
    desc = (desc - l_desc) / (u_desc - l_desc)  # normalize input descriptors

    # % training the network
    x_train = desc[active_mask]
    x_inactive = desc[inactive_mask]

    net = optimize_options["net"]

    # % Change the mapping to trigger distributed control to get distributed gradients
    wrap_options.optimize.mapping = "distributed"

    net._fit_d2p(
        x_train,
        active_mask,
        model,
        wrap_options,
        wrap_returns,
        optimizer,
        optimize_options["parameters"],
        optimize_options["learning_rate"],
        optimize_options["random_state"],
        optimize_options["termination_crit"]["epochs"],
        optimize_options["termination_crit"]["early_stopping"],
        common_options["verbose"],
    )

    # % Manually deallocate control once fit_d2p done
    model._parameters.control.dealloc()

    # % Reset mapping to ann once fit_d2p done
    wrap_options.optimize.mapping = "ann"

    # % predicting at inactive cells
    y = net._predict(x_inactive)

    for i, name in enumerate(optimize_options["parameters"]):
        if name in model.opr_parameters.keys:
            ind = np.argwhere(model.opr_parameters.keys == name).item()

            model.opr_parameters.values[..., ind][inactive_mask] = y[:, i]

        else:
            ind = np.argwhere(model.opr_initial_states.keys == name).item()

            model.opr_inital_states.values[..., ind][inactive_mask] = y[:, i]


def multiple_optimize(
    model: Model,
    samples: Samples,
    mapping: str = "uniform",
    optimizer: str | None = None,
    optimize_options: dict | None = None,
    cost_options: dict | None = None,
    common_options: dict | None = None,
) -> MultipleOptimize:
    
    """
    Optimize the Model on multiple sets of operator parameters or/and initial states.

    TODO: Fill

    Returns
    -------
    mopt : MultipleOptimize
        The multiple optimize results represented as a `MultipleOptimize` object.
    """

    args_options = [
        deepcopy(arg) for arg in [optimize_options, cost_options, common_options]
    ]

    args = _standardize_multiple_optimize_args(
        model,
        samples,
        mapping,
        optimizer,
        *args_options,
    )

    res = _multiple_optimize(model, *args)

    # % Backup cost options info
    res["_cost_options"] = cost_options

    return MultipleOptimize(res)


def _multiple_optimize(
    model: Model,
    samples: Samples,
    mapping: str,
    optimizer: str,
    optimize_options: dict,
    cost_options: dict,
    common_options: dict,
) -> dict:
    if common_options["verbose"]:
        print("</> Multiple Optimize")

    wrap_options = OptionsDT(
        model.setup,
        model.mesh,
        cost_options["njoc"],
        cost_options["njrc"],
    )

    # % Map optimize_options dict to derived type
    for key, value in optimize_options.items():
        if hasattr(wrap_options.optimize, key):
            setattr(wrap_options.optimize, key, value)

        if key == "termination_crit":
            for key_termination_crit, value_termination_crit in value.items():
                if hasattr(wrap_options.optimize, key_termination_crit):
                    setattr(
                        wrap_options.optimize,
                        key_termination_crit,
                        value_termination_crit,
                    )

    # % Map cost_options dict to derived type
    for key, value in cost_options.items():
        if hasattr(wrap_options.cost, key):
            setattr(wrap_options.cost, key, value)

    # % Map common_options dict to derived type
    for key, value in common_options.items():
        if hasattr(wrap_options.comm, key):
            setattr(wrap_options.comm, key, value)

    # % Generate samples info
    nv = samples._problem["num_vars"]
    samples_kind = np.zeros(shape=nv, dtype=np.int32, order="F")
    samples_ind = np.zeros(shape=nv, dtype=np.int32, order="F")

    for i, name in enumerate(samples._problem["names"]):
        if name in model._parameters.opr_parameters.keys:
            samples_kind[i] = 0
            # % Adding 1 because Fortran uses one based indexing
            samples_ind[i] = (
                np.argwhere(model._parameters.opr_parameters.keys == name).item() + 1
            )
        elif name in model._parameters.opr_initial_states.keys:
            samples_kind[i] = 1
            # % Adding 1 because Fortran uses one based indexing
            samples_ind[i] = (
                np.argwhere(model._parameters.opr_initial_states.keys == name).item()
                + 1
            )
        # % Should be unreachable
        else:
            pass

    # % Initialise results
    cost = np.zeros(shape=samples.n_sample, dtype=np.float32, order="F")
    q = np.zeros(
        shape=(*model.obs_response.q.shape, samples.n_sample),
        dtype=np.float32,
        order="F",
    )
    # % Only work with grids (might be changed)
    parameters = np.zeros(
        shape=(
            *model.mesh.flwdir.shape,
            len(optimize_options["parameters"]),
            samples.n_sample,
        ),
        dtype=np.float32,
        order="F",
    )

    wrap_multiple_optimize(
        model.setup,
        model.mesh,
        model._input_data,
        model._parameters,
        model._output,
        wrap_options,
        samples.to_numpy(),
        samples_kind,
        samples_ind,
        cost,
        q,
        parameters,
    )

    # % Finalize parameters and samples for returns
    parameters = dict(
        zip(
            optimize_options["parameters"],
            np.transpose(parameters, (2, 0, 1, 3)),
        )
    )

    for sp in samples._problem[
        "names"
    ]:  # add uncalibrated parameters from samples to parameters
        if not sp in optimize_options["parameters"]:
            value = getattr(samples, sp)
            value = np.tile(value, (*model.mesh.flwdir.shape, 1))

            parameters.update({sp: value})

    samples_fnl = deepcopy(
        samples
    )  # make a deepcopy of samples (will be modified by setattr)

    for op in optimize_options[
        "parameters"
    ]:  # add calibrated paramters from parameters to samples
        if not op in samples._problem["names"]:
            if op in model.opr_parameters.keys:
                value = model.get_opr_parameters(op)[0, 0]

            elif op in model.opr_initial_states.keys:
                value = model.get_opr_initial_states(op)[0, 0]

            # % In case we have other kind of parameters. Should be unreachable.
            else:
                pass

            setattr(samples_fnl, op, value * np.ones(samples.n_sample))
            setattr(samples_fnl, "_dst_" + op, np.ones(samples.n_sample))

    return {
        "cost": cost,
        "q": q,
        "parameters": parameters,
        "_samples": samples_fnl,
    }
