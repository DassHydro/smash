from __future__ import annotations

from smash.fcore._mw_optimize import optimize as wrap_optimize
from smash.fcore._mwd_options import OptionsDT
from smash.fcore._mwd_returns import ReturnsDT

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model

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

    wrap_optimize(
        model.setup,
        model.mesh,
        model._input_data,
        model._parameters,
        model._output,
        wrap_options,
        wrap_returns,
    )
