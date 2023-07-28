from __future__ import annotations

from smash.core.simulation._standardize import (
    _standardize_simulation_mapping,
    _standardize_simulation_optimizer,
    _standardize_simulation_cost_variant,
    _standardize_simulation_optimize_options,
    _standardize_simulation_cost_options,
    _standardize_simulation_common_options,
    _standardize_simulation_parameters_feasibility,
    _standardize_simulation_optimize_options_finalize,
    _standardize_simulation_cost_options_finalize,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash._typing import AnyTuple


def _standardize_optimize_args(
    model: Model,
    mapping: str,
    cost_variant: str,
    optimizer: str | None,
    optimize_options: dict | None,
    cost_options: dict | None,
    common_options: dict | None,
) -> AnyTuple:
    mapping = _standardize_simulation_mapping(mapping)

    cost_variant = _standardize_simulation_cost_variant(cost_variant)

    optimizer = _standardize_simulation_optimizer(mapping, optimizer)

    optimize_options = _standardize_simulation_optimize_options(
        model, mapping, optimizer, optimize_options
    )

    cost_options = _standardize_simulation_cost_options(
        model, cost_variant, cost_options
    )

    common_options = _standardize_simulation_common_options(common_options)

    # % In case model.set_opr_parameters or model.set_opr_initial_states were not used
    _standardize_simulation_parameters_feasibility(model)

    # % Finalize optimize options
    _standardize_simulation_optimize_options_finalize(
        model, mapping, optimizer, optimize_options
    )

    # % Finalize cost_options
    _standardize_simulation_cost_options_finalize(model, cost_variant, cost_options)

    return (
        mapping,
        cost_variant,
        optimizer,
        optimize_options,
        cost_options,
        common_options,
    )
