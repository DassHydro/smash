from __future__ import annotations

from smash.fcore._mw_forward import forward_run as wrap_forward_run
from smash.fcore._mwd_options import OptionsDT
from smash.fcore._mwd_returns import ReturnsDT

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model

__all__ = ["forward_run"]


def forward_run(
    model: Model,
    cost_variant: str = "cls",
    cost_options: dict | None = None,
    common_options: dict | None = None,
) -> Model:
    wmodel = model.copy()

    wmodel.forward_run(cost_variant, cost_options, common_options)

    return wmodel


def _forward_run(
    model: Model, cost_variant: str, cost_options: dict, common_options: dict
):
    wrap_options = OptionsDT(
        model.setup,
        model.mesh,
        cost_options["njoc"],
        cost_options["njrc"],
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

    wrap_forward_run(
        model.setup,
        model.mesh,
        model._input_data,
        model._parameters,
        model._output,
        wrap_options,
        wrap_returns,
    )
