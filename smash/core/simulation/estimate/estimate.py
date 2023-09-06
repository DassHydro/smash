from __future__ import annotations

from smash.core.simulation.estimate._tools import (
    _compute_density,
    _forward_run_with_estimated_parameters,
    _lcurve_forward_run_with_estimated_parameters,
)

from smash.core.simulation.optimize.optimize import MultipleOptimize
from smash.core.simulation.run.run import MultipleForwardRun

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash._typing import Numeric, ListLike
    from smash.core.model.model import Model

__all__ = ["multiset_estimate"]


def multiset_estimate(
    model: Model,
    multiset: MultipleForwardRun | MultipleOptimize,
    alpha: Numeric | ListLike | None = None,
    common_options: dict | None = None,
):
    
    """
    Model assimilation using a Bayesian-like estimation method with multiple sets of operator parameters or/and initial states.

    TODO: Fill

    Returns
    -------
    ret_model : Model
        The optimized Model.
    """

    wmodel = model.copy()

    wmodel.multiset_estimate(multiset, alpha, common_options)

    return wmodel


def _multiset_estimate(
    model: Model,
    multiset: MultipleForwardRun | MultipleOptimize,
    alpha: float | np.ndarray,
    common_options: dict,
) -> dict:
    # TODO: Enhance verbose
    if common_options["verbose"]:
        print("</> Multiple Set Estimate")

    # % Prepare data
    if isinstance(multiset, MultipleForwardRun):
        optimized_parameters = {p: None for p in multiset._samples._problem["names"]}

        prior_data = dict(
            zip(
                optimized_parameters.keys(),
                [
                    np.tile(
                        getattr(multiset._samples, p), (*model.mesh.flwdir.shape, 1)
                    )
                    for p in optimized_parameters.keys()
                ],
            )
        )

    elif isinstance(multiset, MultipleOptimize):
        optimized_parameters = multiset.parameters

        prior_data = multiset.parameters

    else:  # In case we have other kind of multiset. Should be unreachable
        pass

    # % Compute density
    density = _compute_density(
        multiset._samples, optimized_parameters, model.mesh.active_cell
    )

    # % Multiple set estimate
    if isinstance(alpha, float):
        estimator = _forward_run_with_estimated_parameters

    elif isinstance(alpha, np.ndarray):
        estimator = _lcurve_forward_run_with_estimated_parameters

    else:  # Should be unreachable
        pass

    # TODO: add return options here
    estimator(
        alpha,
        model,
        multiset._cost_options,
        prior_data,
        density,
        multiset.cost,
        common_options["ncpu"],
    )
