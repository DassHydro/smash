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

    Parameters
    ----------
    multiset : MultipleForwardRun or MultipleOptimize
        The returned object created by the `smash.multiple_forward_run` or `smash.multiple_optimize` method containing information about multiple sets of operator parameters or initial states.

    alpha : Numeric, ListLike, or None, default None
        A regularization parameter that controls the decay rate of the likelihood function. If **alpha** is a list-like object, the L-curve approach will be used to find an optimal value for the regularization parameter.

        .. note:: If not given, a default numeric range will be set for optimization through the L-curve process.

    common_options : dict or None, default None
        Dictionary containing common options with two elements:

        verbose : bool, default False
            Whether to display information about the running method.

        ncpu : bool, default 1
            Whether to perform a parallel computation.

        .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

    Returns
    -------
    ret_model : Model
        The optimized Model.

    Examples
    --------
    TODO: Fill

    See Also
    --------
    Model.multiset_estimate : Model assimilation using a Bayesian-like estimation method with multiple sets of operator parameters or/and initial states.
    MultipleForwardRun : Represents multiple forward run computation result.
    MultipleOptimize : Represents multiple optimize computation result.
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
