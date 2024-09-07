from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from smash._constant import MAPPING
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

    if setup.n_layers > 0 and optimizer == "sbs":
        warnings.warn(
            f"The SBS optimizer may not be suitable for the {setup.hydrological_module} module", stacklevel=2
        )

    return optimizer


def _standardize_optimize_args(
    model: Model,
    mapping: str,
    optimizer: str | None,
    optimize_options: dict | None,
    cost_options: dict | None,
    common_options: dict | None,
    return_options: dict | None,
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

    return (
        mapping,
        optimizer,
        optimize_options,
        cost_options,
        common_options,
        return_options,
    )


def _standardize_bayesian_optimize_args(
    model: Model,
    mapping: str,
    optimizer: str | None,
    optimize_options: dict | None,
    cost_options: dict | None,
    common_options: dict | None,
    return_options: dict | None,
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

    return (
        mapping,
        optimizer,
        optimize_options,
        cost_options,
        common_options,
        return_options,
    )
