from __future__ import annotations

from smash.core.simulation.optimize.optimize import MultipleOptimize
from smash.core.simulation.run.run import MultipleForwardRun

from smash.core.simulation._standardize import (
    _standardize_simulation_common_options,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash._typing import Numeric, ListLike, AnyTuple

import numpy as np


def _standardize_multiset_estimate_args(
    multiset: MultipleForwardRun | MultipleOptimize,
    alpha: Numeric | ListLike,
    common_options: dict | None,
) -> AnyTuple:
    multiset = _standardize_multiset_estimate_multiset(multiset)

    alpha = _standardize_multiset_estimate_alpha(alpha)

    common_options = _standardize_simulation_common_options(common_options)

    return (multiset, alpha, common_options)


def _standardize_multiset_estimate_multiset(
    multiset: MultipleForwardRun | MultipleOptimize,
) -> MultipleForwardRun | MultipleOptimize:
    if not isinstance(multiset, (MultipleForwardRun, MultipleOptimize)):
        raise TypeError(
            "multiset must be a MultipleForwardRun or MultipleOptimize object"
        )

    return multiset


def _standardize_multiset_estimate_alpha(
    alpha: Numeric | ListLike | None,
) -> float | np.ndarray:
    if alpha is None:
        alpha = np.linspace(-2, 10, 50)
    
    else:
        if isinstance(alpha, (int, float, list, tuple, np.ndarray)):
            alpha = np.array(alpha, ndmin=0)

            if not (alpha.dtype == int or alpha.dtype == float):
                raise TypeError("All values in alpha must be of Numeric type (int, float)")

            if alpha.ndim > 1:
                raise ValueError(f"alpha must be a one-dimensional list-like array")

            else:
                if alpha.size == 1:
                    alpha = float(alpha)

        else:
            raise TypeError(
                "alpha must be a Numeric or ListLike type (List, Tuple, np.ndarray)"
            )

    return alpha
