from __future__ import annotations

from typing import TYPE_CHECKING

from smash.core.simulation._standardize import (
    _standardize_simulation_common_options,
    _standardize_simulation_return_options,
    _standardize_simulation_return_options_finalize,
)
from smash.core.simulation.run.run import MultipleForwardRun

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.util._typing import AnyTuple, ListLike, Numeric

import numpy as np


def _standardize_multiset_estimate_args(
    model: Model,
    multiset: MultipleForwardRun,
    alpha: Numeric | ListLike,
    common_options: dict | None,
    return_options: dict | None,
) -> AnyTuple:
    multiset = _standardize_multiset_estimate_multiset(multiset)

    alpha = _standardize_multiset_estimate_alpha(alpha)

    common_options = _standardize_simulation_common_options(common_options)

    return_options = _standardize_simulation_return_options(model, "multiset_estimate", return_options)

    # % Finalize return_options
    _standardize_simulation_return_options_finalize(model, return_options)

    return (multiset, alpha, common_options, return_options)


def _standardize_multiset_estimate_multiset(
    multiset: MultipleForwardRun,
) -> MultipleForwardRun:
    if not isinstance(multiset, MultipleForwardRun):
        raise TypeError("multiset must be a MultipleForwardRun object")

    return multiset


def _standardize_multiset_estimate_alpha(
    alpha: Numeric | ListLike | None,
) -> float | np.ndarray:
    if alpha is None:
        alpha = np.linspace(-2, 10, 50)

    else:
        if isinstance(alpha, (int, float, list, tuple, np.ndarray)):
            alpha = np.array(alpha, ndmin=0)

            if not (np.issubdtype(alpha.dtype, np.number)):
                raise TypeError("All values in alpha must be of Numeric type (int, float)")

            if alpha.ndim > 1:
                raise ValueError("alpha must be a one-dimensional list-like array")

            else:
                if alpha.size == 1:
                    alpha = float(alpha)

        else:
            raise TypeError("alpha must be a Numeric or ListLike type (List, Tuple, np.ndarray)")

    return alpha
