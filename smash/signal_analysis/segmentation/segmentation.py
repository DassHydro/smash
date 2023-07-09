from __future__ import annotations

from smash.signal_analysis.segmentation._tools import (
    _get_season,
    _missing_values,
    _events_grad,
)

from smash._constant import PEAK_QUANT, MAX_DURATION

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

import numpy as np
import pandas as pd
import warnings


def hydrograph_segmentation(
    model: Model, peak_quant: float = PEAK_QUANT, max_duration: float = MAX_DURATION
):
    """
    Compute segmentation information of flood events over all catchments of the Model.

    .. hint::
        See the :ref:`User Guide <user_guide.in_depth.hydrograph_segmentation>` and :ref:`Math / Num Documentation <math_num_documentation.signal_analysis.hydrograph_segmentation>` for more.

    Parameters
    ----------
    model: Model
        Model object.

    peak_quant: float, default 0.995
        Events will be selected if their discharge peaks exceed the **peak_quant**-quantile of the observed discharge timeseries.

    max_duration: float, default 240
        The expected maximum duration of an event (in hours). If multiple events are detected, their duration may exceed this value.

    Returns
    -------
    res : pandas.DataFrame
        Flood events information obtained from segmentation algorithm.
        The dataframe has 6 columns which are

        - 'code' : the catchment code.
        - 'start' : the beginning of event.
        - 'end' : the end of event.
        - 'maxrainfall' : the moment that the maximum precipation is observed.
        - 'flood' : the moment that the maximum discharge is observed.
        - 'season' : the season that event occurrs.

    Examples
    --------
    >>> import smash
    >>> from smash.factory import load_dataset
    >>> from smash.signal_analysis import hydrograph_segmentation
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Perform segmentation algorithm and display flood events infomation:

    >>> res = hydrograph_segmentation(model)
    >>> res
            code               start                   flood  season
    0  V3524010 2014-11-03 03:00:00 ... 2014-11-04 19:00:00  autumn
    1  V3515010 2014-11-03 10:00:00 ... 2014-11-04 20:00:00  autumn
    2  V3517010 2014-11-03 08:00:00 ... 2014-11-04 16:00:00  autumn

    [3 rows x 6 columns]

    """

    date_range = pd.date_range(
        start=model.setup.start_time,
        periods=model.obs_response.q.shape[1],
        freq=f"{int(model.setup.dt)}s",
    )

    col_name = ["code", "start", "end", "maxrainfall", "flood", "season"]

    df = pd.DataFrame(columns=col_name)

    for i, catchment in enumerate(model.mesh.code):
        prcp_tmp, qobs_tmp, ratio = _missing_values(
            model.atmos_data.mean_prcp[i, :], model.obs_response.q[i, :]
        )

        if prcp_tmp is None:
            warnings.warn(
                f"Reject data at catchment {catchment} ({round(ratio * 100, 2)}% of missing values)"
            )

            pdrow = pd.DataFrame(
                [[catchment] + [np.nan] * (len(col_name) - 1)], columns=col_name
            )
            df = pd.concat([df, pdrow], ignore_index=True)

        else:
            list_events = _events_grad(
                prcp_tmp, qobs_tmp, peak_quant, max_duration, model.setup.dt
            )

            for t in list_events:
                ts = date_range[t["start"]]
                te = date_range[t["end"]]
                peakq = date_range[t["peakQ"]]
                peakp = date_range[t["peakP"]]
                season = _get_season(ts)

                pdrow = pd.DataFrame(
                    [[catchment, ts, te, peakp, peakq, season]], columns=col_name
                )
                df = pd.concat([df, pdrow], ignore_index=True)

    return df
