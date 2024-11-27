from __future__ import annotations

from typing import TYPE_CHECKING

from smash.core.simulation._doc import (
    _default_bayesian_optimize_options_doc_appender,
    _default_optimize_options_doc_appender,
    _smash_default_bayesian_optimize_options_doc_substitution,
    _smash_default_optimize_options_doc_substitution,
)
from smash.core.simulation._standardize import (
    _standardize_default_bayesian_optimize_options_args,
    _standardize_default_optimize_options_args,
    _standardize_simulation_optimize_options,
)

if TYPE_CHECKING:
    from typing import Any

    from smash.core.model.model import Model

__all__ = ["default_bayesian_optimize_options", "default_optimize_options"]


@_smash_default_optimize_options_doc_substitution
@_default_optimize_options_doc_appender
def default_optimize_options(
    model: Model,
    mapping: str = "uniform",
    optimizer: str | None = None,
) -> dict[str, Any]:
    mapping, optimizer = _standardize_default_optimize_options_args(mapping, optimizer)

    return _standardize_simulation_optimize_options(model, "optimize", mapping, optimizer, None)


@_smash_default_bayesian_optimize_options_doc_substitution
@_default_bayesian_optimize_options_doc_appender
def default_bayesian_optimize_options(
    model: Model,
    mapping: str = "uniform",
    optimizer: str | None = None,
) -> dict:
    mapping, optimizer = _standardize_default_bayesian_optimize_options_args(mapping, optimizer)

    return _standardize_simulation_optimize_options(model, "bayesian_optimize", mapping, optimizer, None)
