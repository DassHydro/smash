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
    model: Model,
    multiset: MultipleForwardRun | MultipleOptimize,
    alpha: Numeric | ListLike,
    common_options: dict | None,
) -> AnyTuple:
    multiset = _standardize_multiset_estimate_multiset(model, multiset)

    alpha = _standardize_multiset_estimate_alpha(alpha)

    common_options = _standardize_simulation_common_options(common_options)

    return (multiset, alpha, common_options)


def _standardize_multiset_estimate_multiset(
    model: Model, multiset: MultipleForwardRun | MultipleOptimize
) -> MultipleForwardRun | MultipleOptimize:
    if isinstance(multiset, MultipleForwardRun):
        pass

    elif isinstance(multiset, MultipleOptimize):
        sampl_param = multiset._samples._problem["names"]
        optim_param = multiset.optimized_parameters.keys()

        for sp in sampl_param:
            if not sp in optim_param:
                value = getattr(multiset._samples, sp)
                value = np.tile(value, (*model.mesh.flwdir.shape, 1))

                multiset.optimized_parameters.update({sp: value})

        for op in optim_param:
            if not op in sampl_param:
                if op in model.opr_parameters.keys:
                    value = model.get_opr_parameters(op)[0, 0]

                elif op in model.opr_initial_states.keys:
                    value = model.get_opr_initial_states(op)[0, 0]

                # % In case we have other kind of parameters. Should be unreachable.
                else:
                    pass

                setattr(
                    multiset._samples, op, value * np.ones(multiset._samples.n_sample)
                )
                setattr(
                    multiset._samples, "_" + op, np.ones(multiset._samples.n_sample)
                )

    else:
        raise TypeError(
            "multiset must be a MultipleForwardRun or MultipleOptimize object"
        )

    return multiset


def _standardize_multiset_estimate_alpha(
    alpha: Numeric | ListLike,
) -> float | np.ndarray:
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
