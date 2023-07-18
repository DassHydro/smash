from __future__ import annotations

from smash.core.signal_analysis.metrics._standardize import _standardize_metrics_args

from smash.fcore._mwd_metrics import nse, nnse, kge, mae, mape, mse, rmse, lgrm

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from pandas import Timestamp


__all__ = ["metrics"]


def metrics(
    model: Model, metric: str = "nse", end_warmup: str | Timestamp | None = None
):
    """
    Compute the efficiency/error metrics of the Model based on observed and simulated discharges for each gauge in a multi-catchment hydrological model evaluation.

    Parameters
    ----------
    model : Model
        Model object

    metric : str, default 'nse'
        The efficiency or error criterion. Should be one of

        - 'NSE': Nash-Sutcliffe Efficiency
        - 'NNSE': Normalized Nash-Sutcliffe Efficiency
        - 'KGE': Kling-Gupta Efficiency
        - 'MAE': Mean Absolute Error
        - 'MAPE': Mean Absolute Percentage Error
        - 'MSE': Mean Squared Error
        - 'RMSE': Root Mean Square Error
        - 'LGRM': Logarithmic

    end_warmup : str, pandas.Timestamp or None, default None
        The end of the warm-up period. Evaluation metrics will only be calculated between the end of the warm-up period
        and the end time. The value can be a string that can be interpreted as
        pandas.Timestamp `(see here) <https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.html>`__.
        The **end_warmup** date value must be between the start time and the end time defined in the Model setup.

        .. note::
            If not given, the metrics will be computed for the entire period.

    Returns
    -------
    res : np.ndarray
        A 1D-array of shape (n,) representing the computed efficiency score(s) for **n** catchment(s).

    Examples
    --------

    """
    metric, end_warmup = _standardize_metrics_args(metric, end_warmup, model.setup)

    obs = model.obs_response.q[..., end_warmup:]

    sim = model.sim_response.q[..., end_warmup:]

    ng = obs.shape[0]

    metric_scores = np.zeros(ng)

    for i in range(ng):
        metric_scores[i] = eval(metric)(obs[i], sim[i])

    return metric_scores
