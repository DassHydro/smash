from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from smash.core.simulation._doc import (
    _bayesian_optimize_control_info_doc_appender,
    _optimize_control_info_doc_appender,
    _smash_bayesian_optimize_control_info_doc_substitution,
    _smash_optimize_control_info_doc_substitution,
)
from smash.core.simulation.optimize._standardize import (
    _standardize_bayesian_optimize_args,
    _standardize_optimize_args,
)
from smash.core.simulation.optimize._tools import _get_control_info

if TYPE_CHECKING:
    from typing import Any

    from smash.core.model.model import Model

__all__ = ["bayesian_optimize_control_info", "optimize_control_info"]


@_smash_optimize_control_info_doc_substitution
@_optimize_control_info_doc_appender
def optimize_control_info(
    model: Model,
    mapping: str = "uniform",
    optimizer: str | None = None,
    optimize_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    optimize_options = deepcopy(optimize_options)

    # % Only get mapping, optimizer, optimize_options and cost_options
    *args, _, _, _ = _standardize_optimize_args(
        model,
        mapping,
        optimizer,
        optimize_options,
        None,  # cost_options
        None,  # common_options
        None,  # return_options
        None,  # callback
    )

    return _get_control_info(model, *args)


@_smash_bayesian_optimize_control_info_doc_substitution
@_bayesian_optimize_control_info_doc_appender
def bayesian_optimize_control_info(
    model: Model,
    mapping: str = "uniform",
    optimizer: str | None = None,
    optimize_options: dict[str, Any] | None = None,
    cost_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    args_options = [deepcopy(arg) for arg in [optimize_options, cost_options]]

    # % Only get mapping, optimizer, optimize_options and cost_options
    *args, _, _, _ = _standardize_bayesian_optimize_args(
        model,
        mapping,
        optimizer,
        *args_options,
        None,  # common_options
        None,  # return_options
        None,  # callback
    )

    return _get_control_info(model, *args)
