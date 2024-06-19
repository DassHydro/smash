from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from smash._constant import METRICS

if TYPE_CHECKING:
    from smash.fcore._mwd_setup import SetupDT
    from smash.util._typing import AnyTuple
    from smash.util._typing import ListLike


def _standardize_metrics_criteria(criteria: str | ListLike[str]) -> list:
    if isinstance(criteria, str):
        if criteria.lower() not in METRICS:
            raise ValueError(f"Unknown evaluation metric {criteria}. Choices: {METRICS}")

        criteria = [criteria.lower()]

    elif isinstance(criteria, list):
        for crit in criteria:
            if isinstance(crit, str):
                if crit.lower() not in METRICS:
                    raise ValueError(f"Unknown evaluation metric {crit}. Choices: {METRICS}")
            else:
                raise TypeError("criteria must be str or a list of str")

        criteria = [c.lower() for c in criteria]

    else:
        raise TypeError("criteria must be str or a list of str")

    return criteria


def _standardize_metrics_start_end_eval(eval: str | pd.Timestamp | None, kind: str, setup: SetupDT) -> int:
    st = pd.Timestamp(setup.start_time)
    et = pd.Timestamp(setup.end_time)

    if eval is None:
        if kind == "start":
            eval = pd.Timestamp(st)
        elif kind == "end":
            eval = pd.Timestamp(et)
        else:  # Should be unreachable
            ...

    else:
        if isinstance(eval, str):
            try:
                eval = pd.Timestamp(eval)

            except Exception:
                raise ValueError(f"{kind}_eval '{eval}' argument is an invalid date") from None

        elif isinstance(eval, pd.Timestamp):
            pass

        else:
            raise TypeError("{kind}_eval argument must be str or pandas.Timestamp object")

        if (eval - st).total_seconds() < 0 or (et - eval).total_seconds() < 0:
            raise ValueError(
                f"{kind}_eval '{eval}' argument must be between start time '{st}' and end time '{et}'"
            )

    eval = int((eval - st).total_seconds() / setup.dt)

    return eval


def _standardize_metrics_args(
    criteria: str | ListLike[str],
    start_eval: str | pd.Timestamp | None,
    end_eval: str | pd.Timestamp | None,
    setup: SetupDT,
) -> AnyTuple:
    criteria = _standardize_metrics_criteria(criteria)
    start_eval = _standardize_metrics_start_end_eval(start_eval, "start", setup)
    end_eval = _standardize_metrics_start_end_eval(end_eval, "end", setup)

    return (criteria, start_eval, end_eval)
