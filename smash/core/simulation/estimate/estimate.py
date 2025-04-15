from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from smash._constant import STRUCTURE_RR_INTERNAL_FLUXES
from smash.core.simulation._doc import (
    _multiset_estimate_doc_appender,
    _smash_multiset_estimate_doc_substitution,
)
from smash.core.simulation.estimate._tools import (
    _compute_density,
    _forward_run_with_estimated_parameters,
    _lcurve_forward_run_with_estimated_parameters,
)

if TYPE_CHECKING:
    from typing import Any

    from smash.core.model.model import Model
    from smash.core.simulation.run.run import MultipleForwardRun
    from smash.util._typing import ListLike, Numeric

__all__ = ["MultisetEstimate", "multiset_estimate"]


class MultisetEstimate:
    """
    Represents multiset estimate optional results.

    Attributes
    ----------
    time_step : `pandas.DatetimeIndex`
        A list of length *n* containing the returned time steps.

    rr_states : `FortranDerivedTypeArray`
        A list of length *n* of `RR_StatesDT <fcore._mwd_rr_states.RR_StatesDT>` for each **time_step**.

    q_domain : `numpy.ndarray`
        An array of shape *(nrow, ncol, n)* representing simulated discharges on the domain for each
        **time_step**.

    internal_fluxes : dict[str, `numpy.ndarray`]
        A dictionary where keys are the names of the internal fluxes and the values are array of
        shape *(nrow, ncol, n)* representing an internal flux on the domain for each **time_step**.

    cost : `float`
        Cost value.

    jobs : `float`
        Cost observation component value.

    lcurve_multiset : `dict[str, Any]`
        A dictionary containing the multiset estimate L-curve data. The elements are:

        alpha : `float` or `numpy.ndarray`
            The value of regularization parameter or the list of regularization parameters to be optimized.

        cost : `float` or `numpy.ndarray`
            The corresponding cost value(s).

        mahal_dist : `float` or `numpy.ndarray`
            The corresponding Mahalanobis distance value(s).

        alpha_opt : `float`
            The optimal value of the regularization parameter.

    Notes
    -----
    The object's available attributes depend on what is requested by the user during a call to
    `multiset_estimate`.

    See Also
    --------
    smash.multiset_estimate : Model assimilation using Bayesian-like estimation on multiple sets of solutions.
    """

    def __init__(self, data: dict[str, Any] | None = None):
        if data is None:
            data = {}

        self.__dict__.update(data)

    def __repr__(self):
        dct = self.__dict__

        if dct.keys():
            m = max(map(len, list(dct.keys()))) + 1
            return "\n".join(
                [k.rjust(m) + ": " + repr(type(v)) for k, v in sorted(dct.items()) if not k.startswith("_")]
            )
        else:
            return self.__class__.__name__ + "()"


@_smash_multiset_estimate_doc_substitution
@_multiset_estimate_doc_appender
def multiset_estimate(
    model: Model,
    multiset: MultipleForwardRun,
    alpha: Numeric | ListLike | None = None,
    common_options: dict[str, Any] | None = None,
    return_options: dict[str, Any] | None = None,
) -> Model | tuple[Model, MultisetEstimate]:
    wmodel = model.copy()

    ret_multiset_estimate = wmodel.multiset_estimate(multiset, alpha, common_options, return_options)

    if ret_multiset_estimate is None:
        return wmodel
    else:
        return (wmodel, ret_multiset_estimate)


def _multiset_estimate(
    model: Model,
    multiset: MultipleForwardRun,
    alpha: float | np.ndarray,
    common_options: dict,
    return_options: dict,
) -> MultisetEstimate | None:
    # TODO: Enhance verbose
    if common_options["verbose"]:
        print("</> Multiple Set Estimate")

    # % Compute density
    density = _compute_density(multiset._samples, multiset._spatialized_samples, model.mesh.active_cell)

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
        multiset._spatialized_samples,
        density,
        multiset.cost,
        multiset._cost_options,
        common_options,
        return_options,
    )

    fret = {} if ret_forward_run is None else ret_forward_run.__dict__
    pyret = {"lcurve_multiset": lcurve_multiset} if "lcurve_multiset" in return_options["keys"] else {}

    ret = {**fret, **pyret}

    if ret:
        if "internal_fluxes" in ret:
            ret["internal_fluxes"] = {
                key: ret["internal_fluxes"][..., i]
                for i, key in enumerate(STRUCTURE_RR_INTERNAL_FLUXES[model.setup.structure])
            }
        return MultisetEstimate(ret)
