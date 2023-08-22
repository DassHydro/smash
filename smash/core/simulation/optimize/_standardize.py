from __future__ import annotations

from smash._constant import (
    SIMULATION_OPTIMIZE_OPTIONS_KEYS,
    DEFAULT_SIMULATION_COST_OPTIONS,
)

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
    from smash.core.model.model import Model
    from smash.factory.net.net import Net
    from smash._typing import AnyTuple, Numeric, ListLike
    from pandas import Timestamp


def _standardize_optimize_args(
    model: Model,
    mapping: str,
    cost_variant: str,
    optimizer: str | None,
    optimize_options: dict | None,
    cost_options: dict | None,
    common_options: dict | None,
) -> AnyTuple:
    # % In case model.set_opr_parameters or model.set_opr_initial_states were not used
    _standardize_simulation_parameters_feasibility(model)

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


def _standardize_optimize_options_args(
    model: Model,
    mapping: str,
    optimizer: str | None,
    parameters: str | ListLike | None,
    bounds: dict | None,
    net: Net | None,
    learning_rate: Numeric | None,
    random_state: Numeric | None,
    control_tfm: str | None,
    descriptor: dict | None,
    termination_crit: dict | None,
) -> dict:
    mapping = _standardize_simulation_mapping(mapping)

    optimizer = _standardize_simulation_optimizer(mapping, optimizer)

    optimize_options = {}

    for key in SIMULATION_OPTIMIZE_OPTIONS_KEYS[(mapping, optimizer)]:
        value = locals().get(key)
        optimize_options.setdefault(key, value)

    optimize_options = _standardize_simulation_optimize_options(
        model, mapping, optimizer, optimize_options
    )

    return optimize_options


def _standardize_cost_options_args(
    model: Model,
    cost_variant: str,
    jobs_cmpt: str | ListLike,
    wjobs_cmpt: str | Numeric | ListLike,
    wjreg: Numeric,
    jreg_cmpt: str | ListLike | None,
    wjreg_cmpt: str | Numeric | ListLike,
    gauge: str | ListLike,
    wgauge: str | Numeric | ListLike,
    event_seg: dict | None,
    end_warmup: str | Timestamp | None,
) -> dict:
    cost_variant = _standardize_simulation_cost_variant(cost_variant)

    cost_options = {}

    for key in DEFAULT_SIMULATION_COST_OPTIONS[cost_variant].keys():
        value = locals().get(key)
        cost_options.setdefault(key, value)

    cost_options = _standardize_simulation_cost_options(
        model, cost_variant, cost_options
    )

    return cost_options
