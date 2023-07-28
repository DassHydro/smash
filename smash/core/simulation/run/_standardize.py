from __future__ import annotations

from smash.core.simulation._standardize import (
    _standardize_simulation_cost_variant,
    _standardize_simulation_cost_options,
    _standardize_simulation_common_options,
    _standardize_simulation_parameters_feasibility,
    _standardize_simulation_cost_options_finalize,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash._typing import AnyTuple


def _standardize_forward_run_args(
    model: Model,
    cost_variant: str,
    cost_options: dict | None,
    common_options: dict | None,
) -> AnyTuple:
    cost_variant = _standardize_simulation_cost_variant(cost_variant)

    cost_options = _standardize_simulation_cost_options(
        model, cost_variant, cost_options
    )

    common_options = _standardize_simulation_common_options(common_options)

    # % In case model.set_opr_parameters or model.set_opr_initial_states were not used
    _standardize_simulation_parameters_feasibility(model)

    # % Finalize cost_options
    _standardize_simulation_cost_options_finalize(model, cost_variant, cost_options)

    return (cost_variant, cost_options, common_options)
