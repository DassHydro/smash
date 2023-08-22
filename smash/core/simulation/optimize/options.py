from __future__ import annotations

from smash.core.simulation.optimize._standardize import (
    _standardize_optimize_options_args,
    _standardize_cost_options_args,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.factory.net.net import Net
    from smash._typing import ListLike, Numeric
    from pandas import Timestamp

__all__ = ["optimize_options", "cost_options"]


# % TODO: add docstring - this function is used for creating and the documentation of optimize_options in model.optimize()
def optimize_options(
    model: Model,
    mapping: str = "uniform",
    optimizer: str | None = None,
    parameters: str | ListLike | None = None,
    bounds: dict | None = None,
    net: Net | None = None,
    learning_rate: Numeric | None = None,
    random_state: Numeric | None = None,
    control_tfm: str | None = None,
    descriptor: dict | None = None,
    termination_crit: dict | None = None,
):
    args = _standardize_optimize_options_args(
        model,
        mapping,
        optimizer,
        parameters,
        bounds,
        net,
        learning_rate,
        random_state,
        control_tfm,
        descriptor,
        termination_crit,
    )

    return args


# % TODO: add docstring - this function is used for creating and the documentation of cost_options in model.optimize()
def cost_options(
    model: Model,
    cost_variant: str = "cls",
    jobs_cmpt: str | ListLike = "nse",
    wjobs_cmpt: str | Numeric | ListLike = "mean",
    wjreg: Numeric = 0,
    jreg_cmpt: str | ListLike | None = None,
    wjreg_cmpt: str | Numeric | ListLike = "mean",
    gauge: str | ListLike = "dws",
    wgauge: str | Numeric | ListLike = "mean",
    event_seg: dict | None = None,
    end_warmup: str | Timestamp | None = None,
) -> dict:
    args = _standardize_cost_options_args(
        model,
        cost_variant,
        jobs_cmpt,
        wjobs_cmpt,
        wjreg,
        jreg_cmpt,
        wjreg_cmpt,
        gauge,
        wgauge,
        event_seg,
        end_warmup,
    )

    return args
