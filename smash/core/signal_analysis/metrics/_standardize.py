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
            raise ValueError(f"Unknown efficiency metric {metric}. Choices: {METRICS}")
    else:
        raise TypeError(f"metric must be str")

    return metric.lower()


def _standardize_metrics_end_warmup(
    end_warmup: str | pd.Timestamp | None, setup: SetupDT
) -> int:
    st = pd.Timestamp(setup.start_time)
    et = pd.Timestamp(setup.end_time)

    if end_warmup is None:
        end_warmup = pd.Timestamp(st)

    else:
        if isinstance(end_warmup, str):
            try:
                end_warmup = pd.Timestamp(end_warmup)

            except:
                raise ValueError(
                    f"end_warmup '{end_warmup}' argument is an invalid date"
                )

        elif isinstance(end_warmup, pd.Timestamp):
            pass

        else:
            raise TypeError(
                f"end_warmup argument must be str or pandas.Timestamp object"
            )

        if (end_warmup - st).total_seconds() < 0 or (
            et - end_warmup
        ).total_seconds() < 0:
            raise ValueError(
                f"end_warmup '{end_warmup}' argument must be between start time '{st}' and end time '{et}'"
            )

    end_warmup = int((end_warmup - st).total_seconds() / setup.dt)

    return end_warmup


def _standardize_metrics_args(
    metric: str, end_warmup: str | pd.Timestamp | None, setup: SetupDT
) -> AnyTuple:
    metric = _standardize_metrics_metric(metric)
    end_warmup = _standardize_metrics_end_warmup(end_warmup, setup)

    return (metric, end_warmup)
