from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from smash._constant import METRICS

if TYPE_CHECKING:
    from smash.fcore._mwd_setup import SetupDT
    from smash.util._typing import AnyTuple, ListLike


def _standardize_evaluation_metric(metric: str | ListLike[str]) -> list:
    if isinstance(metric, str):
        if metric.lower() not in METRICS:
            raise ValueError(f"Unknown evaluation metric {metric}. Choices: {METRICS}")

        metric = [metric.lower()]

    elif isinstance(metric, list):
        for i, mtc in enumerate(metric):
            if isinstance(mtc, str):
                if mtc.lower() not in METRICS:
                    raise ValueError(
                        f"Unknown evaluation metric {mtc} at index {i} in metric. Choices: {METRICS}"
                    )
            else:
                raise TypeError("List of evaluation metrics must contain only str")

        metric = [c.lower() for c in metric]

    else:
        raise TypeError("Evaluation metric must be str or a list of str")

    return metric


def _standardize_evaluation_start_end_eval(eval: str | pd.Timestamp | None, kind: str, setup: SetupDT) -> int:
    st = pd.Timestamp(setup.start_time)
    et = pd.Timestamp(setup.end_time)

    if eval is None:
        eval = pd.Timestamp(getattr(setup, f"{kind}_time"))

    else:
        if isinstance(eval, str):
            try:
                eval = pd.Timestamp(eval)

            except Exception:
                raise ValueError(f"{kind}_eval '{eval}' argument is an invalid date") from None

        elif isinstance(eval, pd.Timestamp):
            pass

        else:
            raise TypeError(f"{kind}_eval argument must be str or pandas.Timestamp object")

        if (eval - st).total_seconds() < 0 or (et - eval).total_seconds() < 0:
            raise ValueError(
                f"{kind}_eval '{eval}' argument must be between start time '{st}' and end time '{et}'"
            )

    eval = int((eval - st).total_seconds() / setup.dt)

    return eval


def _standardize_evaluation_args(
    metric: str | ListLike[str],
    start_eval: str | pd.Timestamp | None,
    end_eval: str | pd.Timestamp | None,
    setup: SetupDT,
) -> AnyTuple:
    metric = _standardize_evaluation_metric(metric)
    start_eval = _standardize_evaluation_start_end_eval(start_eval, "start", setup)
    end_eval = _standardize_evaluation_start_end_eval(end_eval, "end", setup)

    return (metric, start_eval, end_eval)
