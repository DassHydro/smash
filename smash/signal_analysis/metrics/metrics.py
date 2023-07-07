from __future__ import annotations

from smash.solver._mwd_efficiency_metric import nse, kge, se, rmse, logarithmic

from smash.signal_analysis.metrics._standardize import (
    _standardize_arrays,
    _standardize_metric,
)

import numpy as np


def efficiency_score(obs: np.ndarray, sim: np.ndarray, metric: str = "nse"):
    """
    Compute the efficiency score for each gauge in a multi-catchment hydrological model evaluation.

    The function takes two 1D- or 2D-arrays, **obs** and **sim**, representing observed and simulated discharges
    respectively, for multiple catchments. Each row in the arrays corresponds to a different
    catchment, and each column represents a time step.

    ..note ::
        For single catchment evaluations, **obs** and **sim** can be provided as 1D-arrays.

    Parameters
    ----------
    obs : np.ndarray
        A 2D-array of shape (g, n) representing observed time series data for **g** catchments and **n** time steps.

    sim : np.ndarray
        A 2D-array of shape (g, n) representing simulated time series data for **g** catchments and **n** time steps.

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
        A 1D-array of shape (g,) containing the computed metric score for each catchment.

    Examples
    --------

    """
    metric = _standardize_metric(metric)

    obs, sim = _standardize_arrays(obs, sim)

    ng = obs.shape[0]

    metric_scores = np.zeros(ng)

    for i in range(ng):
        metric_scores[i] = 1 - eval(metric)(obs[i], sim[i])

    return metric_scores
