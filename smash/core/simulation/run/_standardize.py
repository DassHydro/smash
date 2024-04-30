from __future__ import annotations

from typing import TYPE_CHECKING

from smash.core.simulation._standardize import (
    _standardize_simulation_common_options,
    _standardize_simulation_cost_options,
    _standardize_simulation_cost_options_finalize,
    _standardize_simulation_parameters_feasibility,
    _standardize_simulation_return_options,
    _standardize_simulation_return_options_finalize,
    _standardize_simulation_samples,
)

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.factory.samples.samples import Samples
    from smash.util._typing import AnyTuple


def _standardize_forward_run_args(
    model: Model,
    cost_options: dict | None,
    common_options: dict | None,
    return_options: dict | None,
) -> AnyTuple:
    # % In case model.set_rr_parameters or model.set_rr_initial_states were not used
    _standardize_simulation_parameters_feasibility(model)

    cost_options = _standardize_simulation_cost_options(model, "forward_run", cost_options)

    # % Finalize cost_options
    _standardize_simulation_cost_options_finalize(model, "forward_run", cost_options)

    common_options = _standardize_simulation_common_options(common_options)

    return_options = _standardize_simulation_return_options(model, "forward_run", return_options)

    # % Finalize return_options
    _standardize_simulation_return_options_finalize(model, return_options)

    return (cost_options, common_options, return_options)


def _standardize_multiple_forward_run_args(
    model: Model,
    samples: Samples,
    cost_options: dict | None,
    common_options: dict | None,
) -> AnyTuple:
    samples = _standardize_simulation_samples(model, samples)

    # % In case model.set_rr_parameters or model.set_rr_initial_states were not used
    _standardize_simulation_parameters_feasibility(model)

    cost_options = _standardize_simulation_cost_options(model, "forward_run", cost_options)

    # % Finalize cost_options
    _standardize_simulation_cost_options_finalize(model, "forward_run", cost_options)

    common_options = _standardize_simulation_common_options(common_options)

    return (samples, cost_options, common_options)
