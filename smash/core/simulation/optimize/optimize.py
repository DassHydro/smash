from __future__ import annotations

from smash.fcore._mw_optimize import optimize as wrap_optimize
from smash.fcore._mwd_options import OptionsDT
from smash.fcore._mwd_returns import ReturnsDT

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model

import numpy as np

__all__ = ["optimize"]


def optimize(
    model: Model,
    mapping: str = "uniform",
    cost_variant: str = "cls",
    optimizer: str | None = None,
    optimize_options: dict | None = None,
    cost_options: dict | None = None,
    common_options: dict | None = None,
):
    wmodel = model.copy()

    wmodel.optimize(
        mapping, cost_variant, optimizer, optimize_options, cost_options, common_options
    )

    return wmodel


def _optimize(
    model: Model,
    mapping: str,
    cost_variant: str,
    optimizer: str,
    optimize_options: dict,
    cost_options: dict,
    common_options: dict,
):
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

    l_desc = model._input_data.physio_data._l_descriptor
    u_desc = model._input_data.physio_data._u_descriptor

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
