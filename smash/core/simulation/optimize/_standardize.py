from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING

from smash._constant import GRADIENT_BASED_OPTIMIZER, GRADIENT_FREE_OPTIMIZER, MAPPING
from smash.core.simulation._standardize import (
    _standardize_simulation_common_options,
    _standardize_simulation_cost_options,
    _standardize_simulation_cost_options_finalize,
    _standardize_simulation_mapping,
    _standardize_simulation_optimize_options,
    _standardize_simulation_optimize_options_finalize,
    _standardize_simulation_optimizer,
    _standardize_simulation_parameters_feasibility,
    _standardize_simulation_return_options,
    _standardize_simulation_return_options_finalize,
)

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.fcore._mwd_setup import SetupDT
    from smash.util._typing import AnyTuple


def _standardize_bayesian_optimize_mapping(mapping: str) -> str:
    avail_mapping = MAPPING.copy()
    avail_mapping.remove("ann")  # cannot perform bayesian optimize with ANN mapping

    if isinstance(mapping, str):
        if mapping.lower() not in avail_mapping:
            raise ValueError(f"Invalid mapping '{mapping}' for bayesian optimize. Choices: {avail_mapping}")
    else:
        raise TypeError("mapping argument must be a str")

    return mapping.lower()


def _standardize_optimize_optimizer(mapping: str, optimizer: str, setup: SetupDT) -> str:
    optimizer = _standardize_simulation_optimizer(mapping, optimizer)

    if setup.n_layers > 0 and optimizer in GRADIENT_FREE_OPTIMIZER:
        warnings.warn(
            f"{optimizer} optimizer may not be suitable for the {setup.hydrological_module} module. "
            f"Other choices might be more appropriate: {GRADIENT_BASED_OPTIMIZER}",
            stacklevel=2,
        )

    return optimizer


def _standardize_optimize_callback(callback: callable | None) -> callable | None:
    if callback is None:
        pass

    elif callable(callback):
        cb_signature = inspect.signature(callback)

        if "iopt" not in cb_signature.parameters:
            raise ValueError("Callback function is required to have an 'iopt' argument")

    else:
        raise TypeError("callback argument must be callable")

    return callback


def _standardize_optimize_args(
    model: Model,
    mapping: str,
    optimizer: str | None,
    optimize_options: dict | None,
    cost_options: dict | None,
    common_options: dict | None,
    return_options: dict | None,
    callback: callable | None,
) -> AnyTuple:
    func_name = "optimize"
    # % In case model.set_rr_parameters or model.set_rr_initial_states were not used
    _standardize_simulation_parameters_feasibility(model)

    mapping = _standardize_simulation_mapping(mapping)

    optimizer = _standardize_optimize_optimizer(mapping, optimizer, model.setup)

    optimize_options = _standardize_simulation_optimize_options(
        model, func_name, mapping, optimizer, optimize_options
    )

    # % Finalize optimize options
    _standardize_simulation_optimize_options_finalize(model, mapping, optimizer, optimize_options)

    cost_options = _standardize_simulation_cost_options(model, func_name, cost_options)

    # % Finalize cost_options
    _standardize_simulation_cost_options_finalize(model, func_name, cost_options)

    common_options = _standardize_simulation_common_options(common_options)

    return_options = _standardize_simulation_return_options(model, func_name, return_options)

    # % Finalize return_options
    _standardize_simulation_return_options_finalize(model, return_options)

    callback = _standardize_optimize_callback(callback)

    return (
        mapping,
        optimizer,
        optimize_options,
        cost_options,
        common_options,
        return_options,
        callback,
    )


def _standardize_bayesian_optimize_args(
    model: Model,
    mapping: str,
    optimizer: str | None,
    optimize_options: dict | None,
    cost_options: dict | None,
    common_options: dict | None,
    return_options: dict | None,
    callback: callable | None,
) -> AnyTuple:
    func_name = "bayesian_optimize"

    # % In case model.set_rr_parameters or model.set_rr_initial_states were not used
    _standardize_simulation_parameters_feasibility(model)

    mapping = _standardize_bayesian_optimize_mapping(mapping)

    optimizer = _standardize_optimize_optimizer(mapping, optimizer, model.setup)

    optimize_options = _standardize_simulation_optimize_options(
        model, func_name, mapping, optimizer, optimize_options
    )

    cost_options = _standardize_simulation_cost_options(model, func_name, cost_options)

    common_options = _standardize_simulation_common_options(common_options)

    return_options = _standardize_simulation_return_options(model, func_name, return_options)

    # % Finalize optimize options
    _standardize_simulation_optimize_options_finalize(model, mapping, optimizer, optimize_options)

    # % Finalize cost_options
    _standardize_simulation_cost_options_finalize(model, func_name, cost_options)

    # % Finalize return_options
    _standardize_simulation_return_options_finalize(model, return_options)

    callback = _standardize_optimize_callback(callback)

    return (
        mapping,
        optimizer,
        optimize_options,
        cost_options,
        common_options,
        return_options,
        callback,
    )
