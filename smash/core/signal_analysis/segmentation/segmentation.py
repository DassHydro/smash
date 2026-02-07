from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from smash._constant import MAX_DURATION, PEAK_QUANT, PEAK_VALUE
from smash.core.signal_analysis.segmentation._standardize import (
    _standardize_hydrograph_segmentation_args,
)
from smash.core.signal_analysis.segmentation._tools import _events_grad, _get_season

if TYPE_CHECKING:
    import numpy as np

    from smash.core.model.model import Model
    from smash.util._typing import ListLike, Numeric


__all__ = ["hydrograph_segmentation"]


def hydrograph_segmentation(
    model: Model,
    gauge: str | ListLike = "all",
    peak_quant: Numeric = PEAK_QUANT,
    peak_value: Numeric = PEAK_VALUE,
    max_duration: Numeric = MAX_DURATION,
    by: str = "obs",
):
    """
    Compute segmentation information of flood events over all catchments of Model.

    Parameters
    ----------
    model : `Model`
        Primary data structure of the hydrological model `smash`.

    gauge : `str` or `list[str, ...]`, default 'all'
        Type of gauge for segmentation. There are two ways to specify it:

        - An alias among ``'all'`` (all gauge codes) or ``'dws'`` (most downstream gauge code(s))
        - A gauge code or any sequence of gauge codes. The gauge code(s) given must belong to the gauge codes
          defined in the `Model.mesh`

        >>> gauge = "dws"
        >>> gauge = "V3524010"
        >>> gauge = ["V3524010", "V3515010"]

    peak_quant : `float` or `int`, default 0.995
        Events will be selected if their discharge peaks exceed the **peak_quant**-quantile of the observed
        or simulated (depending on the value of **by**) discharge time series.

    peak_value : `float` or `int`, default 0
        Events will be selected if their discharge peaks exceed the **peak_value** threshold of the observed
        or simulated (depending on the value of **by**) discharge time series (in m3/s).
        This is a complementary criterion to **peak_quant** even if **peak_quant** is not defined.
        Set **peak_quant** to 0 to disable it and use only **peak_value**.

    max_duration : `float`, default 240
        The expected maximum duration of an event (in hours). If multiple events are detected, their duration
        may exceed this value.

    by : `str`, default 'obs'
        Compute segmentation information based on observed (``'obs'``) or simulated (``'sim'``) discharges.
        A simulation (forward run or optimization) is required to obtain the simulated discharge when **by**
        is ``'sim'``.

    Returns
    -------
    segmentation : `pandas.DataFrame`
        Flood events information obtained from segmentation algorithm.
        The dataframe has 7 columns which are

        - ``'code'`` : the catchment code.
        - ``'start'`` : the beginning of event.
        - ``'end'`` : the end of event.
        - ``'multipeak'`` : whether the event has multiple peaks.
        - ``'maxrainfall'`` : the moment that the maximum precipitation is observed.
        - ``'flood'`` : the moment that the maximum discharge is observed.
        - ``'season'`` : the season in which the event occurs.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Perform segmentation algorithm and display flood events information

    >>> hydro_seg = smash.hydrograph_segmentation(model)
    >>> hydro_seg
           code               start                   flood  season
    0  V3524010 2014-11-03 03:00:00 ... 2014-11-04 19:00:00  autumn
    1  V3515010 2014-11-03 10:00:00 ... 2014-11-04 20:00:00  autumn
    2  V3517010 2014-11-03 08:00:00 ... 2014-11-04 16:00:00  autumn
    [3 rows x 7 columns]

    Access all flood events information for a single gauge

    >>> hydro_seg[hydro_seg["code"] == "V3524010"]
           code               start  ...               flood  season
    0  V3524010 2014-11-03 03:00:00  ... 2014-11-04 19:00:00  autumn
    [1 rows x 7 columns]

    Lower the **peak_quant** to potentially retrieve more events

    >>> hydro_seg = smash.hydrograph_segmentation(model, peak_quant=0.99)
    >>> hydro_seg
           code               start  ...               flood  season
    0  V3524010 2014-10-10 04:00:00  ... 2014-10-13 02:00:00  autumn
    1  V3524010 2014-11-03 03:00:00  ... 2014-11-04 19:00:00  autumn
    2  V3515010 2014-10-10 04:00:00  ... 2014-10-13 00:00:00  autumn
    3  V3515010 2014-11-03 10:00:00  ... 2014-11-04 20:00:00  autumn
    4  V3517010 2014-10-09 15:00:00  ... 2014-10-10 23:00:00  autumn
    5  V3517010 2014-11-03 08:00:00  ... 2014-11-04 16:00:00  autumn
    [6 rows x 7 columns]

    Once again, access all flood events information for a single gauge

    >>> hydro_seg[hydro_seg["code"] == "V3524010"]
           code               start  ...               flood  season
    0  V3524010 2014-10-10 04:00:00  ... 2014-10-13 02:00:00  autumn
    1  V3524010 2014-11-03 03:00:00  ... 2014-11-04 19:00:00  autumn
    [2 rows x 7 columns]
    """

    gauge, peak_quant, peak_value, max_duration, by = _standardize_hydrograph_segmentation_args(
        model, gauge, peak_quant, peak_value, max_duration, by
    )

    return _hydrograph_segmentation(model, gauge, peak_quant, peak_value, max_duration, by)


def _hydrograph_segmentation(
    instance: Model, gauge: np.ndarray, peak_quant: float, peak_value: float, max_duration: float, by: str
):
    date_range = pd.date_range(
        start=instance.setup.start_time,
        periods=instance.atmos_data.mean_prcp.shape[1],
        freq=f"{int(instance.setup.dt)}s",
    )

    col_name = ["code", "start", "end", "multipeak", "maxrainfall", "flood", "season"]

    df = pd.DataFrame(columns=col_name)

    for i, catchment in enumerate(instance.mesh.code):
        if catchment not in gauge:
            continue

        prcp = instance.atmos_data.mean_prcp[i, :].copy()

        suffix = "_data" if by == "obs" else ""
        q = getattr(instance, f"response{suffix}").q[i, :].copy()

        list_events = _events_grad(prcp, q, peak_quant, peak_value, max_duration, instance.setup.dt)

        for t in list_events:
            ts = date_range[t["start"]]
            te = date_range[t["end"]]
            peakq = date_range[t["peakQ"]]
            peakp = date_range[t["peakP"]]
            season = _get_season(ts)

            pdrow = pd.DataFrame(
                [[catchment, ts, te, t["multipeak"], peakp, peakq, season]], columns=col_name
            )
            df = pdrow.copy() if df.empty else pd.concat([df, pdrow], ignore_index=True)

    return df
