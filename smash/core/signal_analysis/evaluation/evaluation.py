from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import smash.fcore._mwd_metrics as wrap_module_metrics
from smash.core.signal_analysis.evaluation._standardize import _standardize_evaluation_args

if TYPE_CHECKING:
    from pandas import Timestamp

    from smash.core.model.model import Model
    from smash.util._typing import ListLike


__all__ = ["evaluation"]


def evaluation(
    model: Model,
    metric: str | ListLike[str] = "nse",
    start_eval: str | Timestamp | None = None,
    end_eval: str | Timestamp | None = None,
):
    """
    Evaluate the Model efficiency/error based on observed and simulated discharges.

    Parameters
    ----------
    model : `Model`
        Primary data structure of the hydrological model `smash`.

    metric : `str` or `list[str]`, default 'nse'
        The efficiency or error metric. Should be one of

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
            If not given, the start of evaluation period will be defined as
            the start time in `Model.setup`.

    end_eval : `str`, `pandas.Timestamp` or None, default None
        The end of the evaluation period.
        The value can be a string that can be interpreted as `pandas.Timestamp`.
        The **end_eval** date value must be between the start time and the end time
        defined in `Model.setup`.

        .. note::
            If not given, the end of evaluation period will be defined as
            the end time in `Model.setup`.

    Returns
    -------
    metrics : `numpy.ndarray`
        An array of shape *(ng, nm)* representing the computed metrics for *ng* gauges
        and *nm* evaluation metric.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Optimize Model

    >>> model.optimize()
    </> Optimize
        At iterate     0    nfg =     1    J = 6.95010e-01    ddx = 0.64
        At iterate     1    nfg =    30    J = 9.84107e-02    ddx = 0.64
        At iterate     2    nfg =    59    J = 4.54087e-02    ddx = 0.32
        At iterate     3    nfg =    88    J = 3.81818e-02    ddx = 0.16
        At iterate     4    nfg =   117    J = 3.73617e-02    ddx = 0.08
        At iterate     5    nfg =   150    J = 3.70873e-02    ddx = 0.02
        At iterate     6    nfg =   183    J = 3.68004e-02    ddx = 0.02
        At iterate     7    nfg =   216    J = 3.67635e-02    ddx = 0.01
        At iterate     8    nfg =   240    J = 3.67277e-02    ddx = 0.01
        CONVERGENCE: DDX < 0.01

    Compute multiple evaluation metrics for all catchments

    >>> smash.evaluation(model, metric=["mae", "mse", "nse", "kge"])
    array([[ 3.16766095, 44.77915192,  0.96327233,  0.94864655],
           [ 1.07711864,  4.36171055,  0.90502125,  0.84566253],
           [ 0.33053449,  0.50542408,  0.84976768,  0.8039571 ]])

    Add start and end evaluation dates

    >>> model.setup.start_time, model.setup.end_time
    ('2014-09-15 00:00', '2014-11-14 00:00')
    >>> start_eval, end_eval = "2014-09-30 00:00", "2014-11-01 00:00"

    Compute the Nash-Sutcliffe Efficiency for the entire period and for the specified evaluation period

    >>> smash.evaluation(model, metric="nse")
    array([[0.96327233],
           [0.90502125],
           [0.84976768]])
    >>> smash.evaluation(model, metric="nse", start_eval=start_eval, end_eval=end_eval)
    array([[0.94048417],
           [0.8667959 ],
           [0.76593578]])
    """
    metric, start_eval, end_eval = _standardize_evaluation_args(metric, start_eval, end_eval, model.setup)

    obs = model.response_data.q[..., start_eval:end_eval]

    sim = model.response.q[..., start_eval:end_eval]

    ng = obs.shape[0]

    nm = len(metric)

    evaluation_metric = np.zeros((ng, nm))
    for j in range(nm):
        evaluation_metric_func = getattr(wrap_module_metrics, metric[j])
        for i in range(ng):
            evaluation_metric[i, j] = evaluation_metric_func(obs[i], sim[i])

    return evaluation_metric
