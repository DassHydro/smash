from __future__ import annotations

from smash.tools._common_function import _check_unknown_options

from smash.signal_analysis.segmentation._tools import (
    _detect_peaks,
    _baseflow_separation,
    _get_season,
    _missing_values,
)

from smash._constant import PEAK_QUANT, MAX_DURATION

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

import numpy as np
import pandas as pd
import warnings


def _events_grad(
    p: np.ndarray,
    q: np.ndarray,
    peak_quant: float,
    max_duration: float,  # in hour
    dt: int,
    rg_quant: float = 0.8,
    coef_re: float = 0.2,
    start_seg: int = 72,  # in hour
    st_power: int = 24,  # in hour
    end_search: int = 48,  # in hour
):
    # % time step conversion
    max_duration = round(max_duration * 3600 / dt)
    start_seg = round(start_seg * 3600 / dt)
    st_power = round(st_power * 3600 / dt)
    end_search = round(end_search * 3600 / dt)

    ind = _detect_peaks(q, mph=np.quantile(q[q > 0], peak_quant))
    list_events = []

    for i_peak in ind:
        p_search = p[range(max(i_peak - start_seg, 0), i_peak)]
        p_search_grad = np.gradient(p_search)

        ind_start = _detect_peaks(
            p_search_grad, mph=np.quantile(p_search_grad, rg_quant)
        )

        if ind_start.size > 1:
            power = np.array(
                [
                    np.linalg.norm(p_search[j - 1 : j + st_power - 1], ord=2)
                    for j in ind_start
                ]
            )

            ind_start = ind_start[np.where(power > coef_re * max(power))[0]]

        elif ind_start.size == 0:
            ind_start = np.append(ind_start, np.argmax(p_search_grad))

        ind_start_minq = ind_start[0]

        start = ind_start_minq + max(i_peak - start_seg, 0)

        peakp = _detect_peaks(p[start:i_peak], mpd=len(p))

        if peakp.size == 0:
            peakp = np.argmax(p[start:i_peak]) + start

        else:
            peakp = peakp[0] + start

        fwindow = min(
            start + max_duration + end_search, q.size
        )  # index for determining the end of dflow windows

        if fwindow <= i_peak:  # reject peak at the last time step
            continue

        qbf = _baseflow_separation(q[i_peak - 1 : fwindow - 1])[0]

        dflow = q[i_peak - 1 : fwindow - 1] - qbf
        dflow_windows = (dflow[i:] for i in range(min(end_search, dflow.size)))
        dflow = np.array([sum(i) for i in zip(*dflow_windows)])

        end = i_peak + np.argmin(dflow)

        if len(list_events) > 0:
            prev_start = list_events[-1]["start"]
            prev_end = list_events[-1]["end"]
            prev_peakq = list_events[-1]["peakQ"]
            prev_peakp = list_events[-1]["peakP"]

            # % merge two events respecting to max duration:
            if max(end, prev_end) <= prev_start + max_duration:
                list_events[-1]["end"] = max(end, prev_end)

                if q[i_peak] > q[prev_peakq]:
                    list_events[-1]["peakQ"] = i_peak

                    if p[peakp] > p[prev_peakp]:
                        list_events[-1]["peakP"] = peakp
                continue

        list_events.append(
            {"start": start, "end": end, "peakP": peakp, "peakQ": i_peak}
        )

    return list_events


def _mask_event(
    model: Model,
    peak_quant: float = PEAK_QUANT,
    max_duration: float = MAX_DURATION,  # in hour
    **unknown_options,
):
    _check_unknown_options("event segmentation", unknown_options)

    mask = np.zeros(model.obs_response.q.shape)

    for i, catchment in enumerate(model.mesh.code):
        prcp_tmp, qobs_tmp, ratio = _missing_values(
            model.atmos_data.mean_prcp[i, :], model.obs_response.q.shape[i, :]
        )

        if prcp_tmp is None:
            warnings.warn(
                f"Reject data at catchment {catchment} ({round(ratio * 100, 2)}% of missing values)"
            )

        else:
            list_events = _events_grad(
                prcp_tmp, qobs_tmp, peak_quant, max_duration, model.setup.dt
            )

            for event_number, t in enumerate(list_events):
                ts = t["start"]
                te = t["end"]

                mask[i, ts : te + 1] = event_number + 1

    return mask


def event_segmentation(
    model: Model, peak_quant: float = PEAK_QUANT, max_duration: float = MAX_DURATION
):
    """
    Compute segmentation information of flood events over all catchments of the Model.

    .. hint::
        See the :ref:`User Guide <user_guide.in_depth.event_segmentation>` and :ref:`Math / Num Documentation <math_num_documentation.signal_analysis.hydrograph_segmentation>` for more.

    Parameters
    ----------
    model: Model
        Model object.

    peak_quant: float, default 0.995
        Events will be selected if their discharge peaks exceed the **peak_quant**-quantile of the observed discharge timeseries.

    max_duration: float, default 240
        The expected maximum duration of an event (in hour).

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
    >>> from smash.signal_analysis import event_segmentation
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Perform segmentation algorithm and display flood events infomation:

    >>> res = event_segmentation(model)
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
