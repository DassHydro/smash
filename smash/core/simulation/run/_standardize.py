from __future__ import annotations

from smash.core.simulation._standardize import (
    _standardize_simulation_samples,
    _standardize_simulation_cost_options,
    _standardize_simulation_common_options,
    _standardize_simulation_parameters_feasibility,
    _standardize_simulation_cost_options_finalize,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.factory.samples.samples import Samples
    from smash._typing import AnyTuple


def _standardize_forward_run_args(
    model: Model,
    cost_options: dict | None,
    common_options: dict | None,
) -> AnyTuple:
    # % In case model.set_opr_parameters or model.set_opr_initial_states were not used
    _standardize_simulation_parameters_feasibility(model)

    cost_options = _standardize_simulation_cost_options(model, cost_options)

    common_options = _standardize_simulation_common_options(common_options)

    # % Finalize cost_options
    _standardize_simulation_cost_options_finalize(model, cost_options)

    return (cost_options, common_options)


def _standardize_multiple_forward_run_args(
    model: Model,
    samples: Samples,
    cost_options: dict | None,
    common_options: dict | None,
) -> AnyTuple:
    samples = _standardize_simulation_samples(model, samples)

    forward_run_args = _standardize_forward_run_args(
        model, cost_options, common_options
    )

    return (samples, *forward_run_args)
