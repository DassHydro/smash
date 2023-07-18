from __future__ import annotations

from smash._constant import METRICS

import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.fcore._mwd_setup import SetupDT
    from smash._typing import AnyTuple


def _standardize_metrics_metric(metric: str) -> str:
    if isinstance(metric, str):
        if metric.lower() not in METRICS:

            raise ValueError(
                f"Unknown efficiency metric {metric}. Choices: {METRICS}"
            )
    else:
        raise TypeError(f"metric must be str")
    
    return metric.lower()
    

def _standardize_metrics_cst(cst: str | pd.Timestamp | None, setup: SetupDT) -> int:
    st = pd.Timestamp(setup.start_time)
    et = pd.Timestamp(setup.end_time)
    
    if cst is None:
        cst = pd.Timestamp(st)

    else:

        if isinstance(cst, str):
            try:
                cst = pd.Timestamp(cst)

            except:
                raise ValueError(f"cst '{cst}' argument is an invalid date")

        elif isinstance(cst, pd.Timestamp):
            pass

        else:
            raise TypeError(f"cst argument must be str or pandas.Timestamp object")

        if (cst - st).total_seconds() < 0 or (et - cst).total_seconds() < 0:
            raise ValueError(
                f"cst '{cst}' argument must be between start time '{st}' and end time '{et}'"
            )
        
    cst = int((cst - st).total_seconds() / setup.dt)

    return cst


def _standardize_metrics_args(metric: str, cst: str | pd.Timestamp | None, setup: SetupDT) -> AnyTuple:
    metric = _standardize_metrics_metric(metric)
    cst = _standardize_metrics_cst(cst, setup)

    return (metric, cst)
