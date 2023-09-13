from __future__ import annotations

from smash._constant import SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS

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

__all__ = ["MultisetEstimate", "multiset_estimate"]


class MultisetEstimate:
    """
    Represents multiset estimate optional results.

    Attributes
    ----------
    time_step : pandas.DatetimeIndex
        A pandas.DatetimeIndex containing *n* returned time steps.

    opr_states : list
        A list of length *n* of Opr_StatesDT for each **time_step**.

    q_domain : numpy.ndarray
        An array of shape *(nrow, ncol, n)* representing simulated discharges on the domain for each **time_step**.

    cost : float
        Cost value.

    jobs : float
        Cost observation component value.

    jreg : float
        Cost regularization component value.

    lcurve_multiset : dict
        A dictionary containing the multiset estimate lcurve data. The elements are:

        mahal_dist : float or numpy.ndarray
            TODO TH: Fill

        cost : float or numpy.ndarray
            TODO TH: Fill

        alpha : float or numpy.ndarray
            TODO TH: Fill

        alpha_opt : float
            TODO TH: Fill

    Notes
    -----
    The object's available attributes depend on what is requested by the user during a call to `smash.multiset_estimate`.

    See Also
    --------
    smash.multiset_estimate : Model assimilation using Bayesian-like estimation on multiple sets of solutions.
    """

    def __init__(self, data: dict | None = None):
        if data is None:
            data = {}

        self.__dict__.update(data)

    def __repr__(self):
        dct = self.__dict__

        if dct.keys():
            m = max(map(len, list(dct.keys()))) + 1
            return "\n".join(
                [
                    k.rjust(m) + ": " + repr(type(v))
                    for k, v in sorted(dct.items())
                    if not k.startswith("_")
                ]
            )
        else:
            return self.__class__.__name__ + "()"


def multiset_estimate(
    model: Model,
    multiset: MultipleForwardRun | MultipleOptimize,
    alpha: Numeric | ListLike | None = None,
    common_options: dict | None = None,
    return_options: dict | None = None,
) -> Model | (Model, MultisetEstimate):
    """
    Model assimilation using Bayesian-like estimation on multiple sets of solutions.

    Parameters
    ----------
    model : Model
        Model object.

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

    return_options : dict or None, default None
        Dictionary containing return options to save intermediate variables. The elements are:

        time_step : str, pandas.Timestamp, pandas.DatetimeIndex or ListLike, default 'all'
            Returned time steps. There are five ways to specify it:

            - A date as a character string which respect pandas.Timestamp format (i.e. '1997-12-21', '19971221', ...).
            - An alias among 'all' (return all time steps).
            - A pandas.Timestamp object.
            - A pandas.DatetimeIndex object.
            - A sequence of dates as character string or pandas.Timestamp (i.e. ['1998-23-05', '1998-23-06'])

            .. note::
                It only applies to the following variables: 'opr_states' and 'q_domain'

        opr_states : bool, default False
            Whether to return operator states for specific time steps.

        q_domain : bool, defaul False
            Whether to return simulated discharge on the whole domain for specific time steps.

        cost : bool, default False
            Whether to return cost value.

        jobs : bool, default False
            Whether to return jobs (observation component of cost) value.

        jreg : bool, default False
            Whether to return jreg (regularization component of cost) value.

        lcurve_multiset : bool, default False
            Whether to return the multiset estimate lcurve.

        .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

    Returns
    -------
    ret_model : Model
        The Model with multiset estimate outputs.

    ret_multiset_estimate : MultisetEstimate or None, default None
        It returns a `smash.MultisetEstimate` object containing the intermediate variables defined in **return_options**. If no intermediate variables are defined, it returns None.

    Examples
    --------
    TODO: Fill

    See Also
    --------
    Model.multiset_estimate : Model assimilation using Bayesian-like estimation on multiple sets of solutions.
    MultisetEstimate : Represents multiset estimate optional results.
    MultipleForwardRun : Represents multiple forward run computation result.
    MultipleOptimize : Represents multiple optimize computation result.
    """

    wmodel = model.copy()

    ret_multiset_estimate = wmodel.multiset_estimate(
        multiset, alpha, common_options, return_options
    )

    if ret_multiset_estimate is None:
        return wmodel
    else:
        return wmodel, ret_multiset_estimate


def _multiset_estimate(
    model: Model,
    multiset: MultipleForwardRun | MultipleOptimize,
    alpha: float | np.ndarray,
    common_options: dict,
    return_options: dict,
) -> MultisetEstimate | None:
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

    ret_forward_run, lcurve_multiset = estimator(
        alpha,
        model,
        prior_data,
        density,
        multiset.cost,
        multiset._cost_options,
        common_options,
        return_options,
    )

    fret = {} if ret_forward_run is None else ret_forward_run.__dict__
    pyret = {}

    if "lcurve_multiset" in return_options["keys"]:
        pyret["lcurve_multiset"] = lcurve_multiset

    ret = {**fret, **pyret}

    if ret:
        return MultisetEstimate(ret)
