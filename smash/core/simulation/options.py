from __future__ import annotations

from smash.core.simulation._standardize import (
    _standardize_simulation_optimize_options,
    _standardize_default_optimize_options_args,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model

__all__ = ["default_optimize_options"]


def default_optimize_options(
    model: Model,
    mapping: str = "uniform",
    optimizer: str | None = None,
) -> dict:
    """
    Get the default optimization options for the Model object.

    Parameters
    ----------
    model : Model
        Model object.

    mapping : str, default 'uniform'
        Type of mapping. Should be one of 'uniform', 'distributed', 'hyper-linear', 'hyper-polynomial', 'ann'.

    optimizer : str or None, default None
        Name of optimizer. Should be one of 'sbs', 'lbfgsb', 'sgd', 'adam', 'adagrad', 'rmsprop'.

        .. note::
            If not given, a default optimizer will be set depending on the optimization mapping:

            - **mapping** = 'uniform'; **optimizer** = 'sbs'
            - **mapping** = 'distributed', 'hyper-linear', or 'hyper-polynomial'; **optimizer** = 'lbfgsb'
            - **mapping** = 'ann'; **optimizer** = 'adam'

    Returns
    -------
    TODO: Fill
    """

    mapping, optimizer = _standardize_default_optimize_options_args(mapping, optimizer)

    return _standardize_simulation_optimize_options(model, mapping, optimizer, None)
