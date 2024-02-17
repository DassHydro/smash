from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import smash.fcore._mwd_metrics as wrap_module_metrics
from smash.core.signal_analysis.metrics._standardize import _standardize_metrics_args

if TYPE_CHECKING:
    from pandas import Timestamp

    from smash.core.model.model import Model


__all__ = ["metrics"]


def metrics(model: Model, metric: str = "nse", end_warmup: str | Timestamp | None = None):
    """
    Compute the efficiency/error metrics of Model based on observed and simulated discharges
    for each gauge in a multi-catchment hydrological model evaluation.

    Parameters
    ----------
    model : `Model <smash.Model>`
        Primary data structure of the hydrological model `smash`.

    metric : `str`, default 'nse'
        The efficiency or error criterion. Should be one of

        - '``nse'``: Nash-Sutcliffe Efficiency
        - '``nnse'``: Normalized Nash-Sutcliffe Efficiency
        - '``kge'``: Kling-Gupta Efficiency
        - '``mae'``: Mean Absolute Error
        - '``mape'``: Mean Absolute Percentage Error
        - '``mse'``: Mean Squared Error
        - '``rmse'``: Root Mean Squared Error
        - '``lgrm'``: Logarithmic

    end_warmup : `str`, `pandas.Timestamp` or None, default None
        The end of the warm-up period. Evaluation metrics will only be calculated between the end of the
        warm-up period and the end time. The value can be a string that can be interpreted as
        `pandas.Timestamp`. The **end_warmup** date value must be between the start time and the end time
        defined in `Model.setup`.

        .. note::
            If not given, the metrics will be computed for the entire period.

    Returns
    -------
    metrics : `numpy.ndarray`
        An array of shape *(n,)* representing the computed evaluation metric for *n* catchment(s).

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Optimize Model

    >>> model.optimize()
    </> Optimize
        At iterate      0    nfg =     1    J =      0.643190    ddx = 0.64
        At iterate      1    nfg =    30    J =      0.097397    ddx = 0.64
        At iterate      2    nfg =    59    J =      0.052158    ddx = 0.32
        At iterate      3    nfg =    88    J =      0.043086    ddx = 0.08
        At iterate      4    nfg =   118    J =      0.040684    ddx = 0.02
        At iterate      5    nfg =   152    J =      0.040604    ddx = 0.01
        CONVERGENCE: DDX < 0.01

    Compute the Mean Squared Error for all catchments

    >>> smash.metrics(model, metric="mse")
    array([49.50750351,  4.31146955,  0.57386672])

    The Kling-Gupta Efficiency

    >>> smash.metrics(model, metric="kge")
    array([0.96001273, 0.86749017, 0.81862521])

    Add end warm-up date (1 month)

    >>> model.setup.start_time, model.setup.end_time
    ('2014-09-15 00:00', '2014-11-14 00:00')
    >>> end_warmup = "2014-10-14 00:00"

    Compute the Nash-Sutcliffe Efficiency with and without warm-up

    >>> smash.metrics(model, metric="nse")
    array([0.95939589, 0.90611488, 0.82942516])
    >>> smash.metrics(model, metric="nse", end_warmup=end_warmup)
    array([0.9774977 , 0.92978942, 0.87382156])
    """
    metric, end_warmup = _standardize_metrics_args(metric, end_warmup, model.setup)

    obs = model.response_data.q[..., end_warmup:]

    sim = model.response.q[..., end_warmup:]

    ng = obs.shape[0]

    evaluation_metric = np.zeros(ng)

    evaluation_metric_func = getattr(wrap_module_metrics, metric)

    for i in range(ng):
        evaluation_metric[i] = evaluation_metric_func(obs[i], sim[i])

    return evaluation_metric
