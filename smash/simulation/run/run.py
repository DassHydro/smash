from __future__ import annotations

from smash.solver._mw_forward import forward_run as fw_run

from smash.simulation._standardize import _standardize_options, _standardize_returns

from smash.simulation.run._standardize import _standardize_parameter

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model
    from smash.solver._mwd_options import OptionsDT
    from smash.solver._mwd_returns import ReturnsDT

__al__ = ["forward_run"]


def forward_run(
    model: Model, options: OptionsDT | None = None, returns: ReturnsDT | None = None
):
    new_model = model.copy()

    _forward_run(new_model, options, returns)

    return new_model


def _forward_run(instance: Model, options: OptionsDT, returns: ReturnsDT):
    options = _standardize_options(options, instance.setup)

    returns = _standardize_returns(returns)

    _standardize_parameter(instance._parameters)

    fw_run(
        instance.setup,
        instance.mesh,
        instance._input_data,
        instance._parameters,
        instance._output,
        options,
        returns,
    )
