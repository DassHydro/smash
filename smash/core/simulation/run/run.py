from __future__ import annotations

from smash.core.simulation._standardize import (
    _standardize_options,
    _standardize_returns,
)

from smash.core.simulation.run._standardize import _standardize_opr_parameter_state

from smash.fcore._mw_forward import forward_run as fw_run

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.fcore._mwd_options import OptionsDT
    from smash.fcore._mwd_returns import ReturnsDT

__all__ = ["forward_run"]


def forward_run(
    model: Model, options: OptionsDT | None = None, returns: ReturnsDT | None = None
):
    new_model = model.copy()

    _forward_run(new_model, options, returns)

    return new_model


def _forward_run(instance: Model, options: OptionsDT, returns: ReturnsDT):
    options = _standardize_options(options, instance.setup)

    returns = _standardize_returns(returns)

    _standardize_opr_parameter_state(instance._parameters)

    fw_run(
        instance.setup,
        instance.mesh,
        instance._input_data,
        instance._parameters,
        instance._output,
        options,
        returns,
    )
