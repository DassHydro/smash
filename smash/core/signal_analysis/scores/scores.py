from __future__ import annotations

from smash.core.signal_analysis.scores._standardize import _standardize_metric

from smash.fcore._mwd_efficiency_metric import nse, kge, se, rmse, logarithmic

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model


__all__ = ["efficiency_score"]


def efficiency_score(model: Model, metric: str = "nse"):
    """
    Compute the efficiency score of the Model based on observed and simulated discharges for each gauge in a multi-catchment hydrological model evaluation.

    Parameters
    ----------
    model : Model
        Model object
        
    metric : str, default 'nse'
        The efficiency metric criterion. Should be one of

        - 'nse'
        - 'kge'
        - 'se'
        - 'rmse'
        - 'logarithmic'

    Returns
    -------
    res : np.ndarray
        A 1D-array of shape (n,) representing the computed efficiency score(s) for **n** catchment(s).

    Examples
    --------

    """
    metric = _standardize_metric(metric)

    obs = model.obs_response.q

    sim = model.sim_response.q

    ng = obs.shape[0]

    metric_scores = np.zeros(ng)

    for i in range(ng):
        metric_scores[i] = 1 - eval(metric)(obs[i], sim[i])

    return metric_scores
