from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import smash.fcore._mwd_metrics as wrap_module_metrics
from smash.core.signal_analysis.metrics._standardize import _standardize_metrics_args

if TYPE_CHECKING:
    from smash.util._typing import ListLike

    from pandas import Timestamp

    from smash.core.model.model import Model


__all__ = ["metrics"]


def metrics(
    model: Model,
    criteria: str | ListLike[str] = "nse",
    start_eval: str | Timestamp | None = None,
    end_eval: str | Timestamp | None = None,
):
    """
    Compute the efficiency/error metrics of Model based on observed and simulated discharges
    for each gauge in a multi-catchment hydrological model evaluation.

    Parameters
    ----------
    model : `Model <smash.Model>`
        Primary data structure of the hydrological model `smash`.

    criteria : `str` or `list[str]`, default 'nse'
        The efficiency or error criteria. Should be one of

        - '``nse'``: Nash-Sutcliffe Efficiency
        - '``nnse'``: Normalized Nash-Sutcliffe Efficiency
        - '``kge'``: Kling-Gupta Efficiency
        - '``mae'``: Mean Absolute Error
        - '``mape'``: Mean Absolute Percentage Error
        - '``mse'``: Mean Squared Error
        - '``rmse'``: Root Mean Squared Error
        - '``lgrm'``: Logarithmic

    start_eval : `str`, `pandas.Timestamp` or None, default None
        The start of the evaluation period. 
        The value can be a string that can be interpreted as `pandas.Timestamp`. 
        The **start_eval** date value must be between the start time and the end time
        defined in `Model.setup`.

        .. note::
            If not given, the start of evaluation period will be defined as the start time in `Model.setup`.

    end_eval : `str`, `pandas.Timestamp` or None, default None
        The end of the evaluation period. 
        The value can be a string that can be interpreted as `pandas.Timestamp`. 
        The **end_eval** date value must be between the start time and the end time
        defined in `Model.setup`.

        .. note::
            If not given, the end of evaluation period will be defined as the end time in `Model.setup`.

    Returns
    -------
    metrics : `numpy.ndarray`
        An array of shape *(m, n)* representing the computed metrics for *m* catchments and *n* evaluation criteria.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Optimize Model

    >>> model.optimize()
    </> Optimize
        At iterate      0    nfg =     1    J =      0.695010    ddx = 0.64
        At iterate      1    nfg =    30    J =      0.098411    ddx = 0.64
        At iterate      2    nfg =    59    J =      0.045409    ddx = 0.32
        At iterate      3    nfg =    88    J =      0.038182    ddx = 0.16
        At iterate      4    nfg =   117    J =      0.037362    ddx = 0.08
        At iterate      5    nfg =   150    J =      0.037087    ddx = 0.02
        At iterate      6    nfg =   183    J =      0.036800    ddx = 0.02
        At iterate      7    nfg =   216    J =      0.036763    ddx = 0.01
        CONVERGENCE: DDX < 0.01

    Compute multiple evaluation metrics for all catchments

    >>> smash.metrics(model, criteria=["mae", "mse", "nse", "kge"])
    array([[ 3.16965151, 44.78328323,  0.96327233,  0.94752783],
           [ 1.07771611,  4.38410997,  0.90453297,  0.84582865],
           [ 0.33045691,  0.50611502,  0.84956211,  0.8045246 ]])

    Add start and end evaluation dates

    >>> model.setup.start_time, model.setup.end_time
    ('2014-09-15 00:00', '2014-11-14 00:00')
    >>> start_eval, end_eval = "2014-09-30 00:00", "2014-11-01 00:00"

    Compute the Nash-Sutcliffe Efficiency for the entire period and for the specified evaluation period

    >>> smash.metrics(model, criteria="nse")
    array([[0.96327233],
           [0.90453297],
           [0.84956211]])
    >>> smash.metrics(model, criteria="nse", start_eval=start_eval, end_eval=end_eval)
    array([[0.9404493 ],
           [0.86493075],
           [0.76471144]])
    """
    criteria, start_eval, end_eval = _standardize_metrics_args(criteria, start_eval, end_eval, model.setup)

    obs = model.response_data.q[..., start_eval:end_eval]

    sim = model.response.q[..., start_eval:end_eval]

    ng = obs.shape[0]

    nm = len(criteria)

    evaluation_metric = np.zeros((ng, nm))
    for j in range(nm):
        evaluation_metric_func = getattr(wrap_module_metrics, criteria[j])
        for i in range(ng):
            evaluation_metric[i, j] = evaluation_metric_func(obs[i], sim[i])

    return evaluation_metric
