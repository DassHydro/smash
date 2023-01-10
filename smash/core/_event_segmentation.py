from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

import numpy as np
import pandas as pd
from datetime import date, datetime
import warnings


def _missing_values(
    p: np.ndarray, q: np.ndarray, keep: float = 0.2, method: str = "nearest"
):
    """Deal with misssing values problem on P and Q"""

    df = pd.DataFrame({"p": p, "q": q})
    df.p = np.where(df.p < 0, np.nan, df.p)
    df.q = np.where(df.q < 0, np.nan, df.q)

    mvrat = max(df.p.isna().sum() / len(p), df.q.isna().sum() / len(q))

    if mvrat <= keep:
        indp = np.where(df.p.to_numpy() >= 0)
        indq = np.where(df.q.to_numpy() >= 0)

        df.p[0] = df.p[indp[0][0]]
        df.p[len(df) - 1] = df.p[indp[0][-1]]

        df.q[0] = df.q[indq[0][0]]
        df.q[len(df) - 1] = df.q[indq[0][-1]]

        df.interpolate(method=method, inplace=True)

        return df.p.to_numpy(), df.q.to_numpy(), mvrat

    else:

        return None, None, mvrat


def _baseflow_separation(
    streamflow: np.ndarray, filter_parameter: float = 0.925, passes: int = 3
):

    n = len(streamflow)
    ends = np.array([n - 1 if i % 2 == 1 else 0 for i in range(passes + 1)])
    addtostart = np.array([-1 if i % 2 == 1 else 1 for i in range(passes)])

    btp = np.copy(streamflow)  #% Previous pass's baseflow approximation
    qft = np.zeros(n)
    bt = np.zeros(n)

    if streamflow[0] < np.quantile(streamflow, 0.25):
        bt[0] = streamflow[0]

    else:
        bt[0] = np.mean(streamflow) / 1.5

    #% Guess baseflow value in first time step.
    for j in range(passes):
        rang = np.linspace(
            ends[j] + addtostart[j],
            ends[j + 1],
            np.abs(ends[j] + addtostart[j] - ends[j + 1]) + 1,
        )
        for ii in rang:

            i = int(ii)

            if (
                filter_parameter * bt[i - addtostart[j]]
                + ((1 - filter_parameter) / 2) * (btp[i] + btp[i - addtostart[j]])
                > btp[i]
            ):
                bt[i] = btp[i]

            else:
                bt[i] = filter_parameter * bt[i - addtostart[j]] + (
                    (1 - filter_parameter) / 2
                ) * (btp[i] + btp[i - addtostart[j]])
            qft[i] = streamflow[i] - bt[i]

        if j < passes - 1:

            btp = np.copy(bt)

            if streamflow[ends[j + 1]] < np.mean(btp):
                bt[ends[j + 1]] = streamflow[ends[j + 1]] / 1.2

            else:
                bt[ends[j + 1]] = np.mean(btp)

    return bt, qft


def _get_season(now: datetime):

    year = 2000  #% dummy leap year to allow input X-02-29 (leap day)
    seasons = [
        ("winter", (date(year, 1, 1), date(year, 3, 20))),
        ("spring", (date(year, 3, 21), date(year, 6, 20))),
        ("summer", (date(year, 6, 21), date(year, 9, 22))),
        ("autumn", (date(year, 9, 23), date(year, 12, 20))),
        ("winter", (date(year, 12, 21), date(year, 12, 31))),
    ]

    if isinstance(now, datetime):
        now = now.date()

    now = now.replace(year=year)

    return next(season for season, (start, end) in seasons if start <= now <= end)


def _detect_peaks(
    x: np.ndarray,
    mph: float | None = None,
    mpd: int = 1,
    threshold: int = 0,
    edge: str = "rising",
    kpsh: bool = False,
    valley: bool = False,
):

    x = np.atleast_1d(x).astype("float64")
    if x.size < 3:
        return np.array([], dtype=int)

    if valley:
        x = -x

    #% find indices of all peaks
    dx = x[1:] - x[:-1]
    #% handle NaN's
    indnan = np.where(np.isnan(x))[0]

    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)

    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]

    else:
        if edge.lower() in ["rising", "both"]:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]

        if edge.lower() in ["falling", "both"]:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]

    ind = np.unique(np.hstack((ine, ire, ife)))

    #% handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[
            np.in1d(
                ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True
            )
        ]

    #% first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]

    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]

    #% remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]

    #% remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])

    #% detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)

        for i in range(ind.size):

            if not idel[i]:
                #% keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) & (
                    x[ind[i]] > x[ind] if kpsh else True
                )
                idel[i] = 0  #% Keep current peak

        #% remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def _events_grad(
    p: np.ndarray,
    q: np.ndarray,
    peak_quant: float,
    max_duration: int,
    rg_quant: float = 0.8,
    coef_re: float = 0.2,
    start_seg: int = 72,
    search_start: int = 12,
    search_end: int = 24,
):

    ind = _detect_peaks(q, mph=np.quantile(q[q > 0], peak_quant))
    list_events = []

    for i in ind:
        p_search = p[range(max(i - start_seg, 0), i)]
        p_search_grad = np.gradient(p_search)

        try:
            ind_start = _detect_peaks(
                p_search_grad, mph=np.quantile(p_search_grad, rg_quant)
            )
        except:
            continue

        power = np.array(
            [
                np.linalg.norm(p_search[j - 1 : j + search_start - 1], ord=2)
                for j in ind_start
            ]
        )

        try:
            ind_start = ind_start[np.where(power > coef_re * max(power))[0]]
        except:
            continue

        try:
            ind_start_minq = ind_start[0]
        except:
            continue

        start = ind_start_minq + max(i - start_seg, 0)

        try:
            peakp = _detect_peaks(p[start:i], mpd=len(p))[0]
        except:
            continue

        peakp += start

        qbf = _baseflow_separation(q[i - 1 : start + max_duration + search_end - 1])[0]

        dflow = q[i - 1 : start + max_duration + search_end - 1] - qbf
        dflow = np.array([sum(i) for i in zip(*(dflow[i:] for i in range(search_end)))])

        end = i + np.argmin(dflow)

        if len(list_events) > 0:
            prev_start = list_events[-1]["start"]
            prev_end = list_events[-1]["end"]
            prev_peakq = list_events[-1]["peakQ"]
            prev_peakp = list_events[-1]["peakP"]

            #% merge two events respecting to max duration:
            if max(end, prev_end) <= prev_start + max_duration:
                list_events[-1]["end"] = max(end, prev_end)

                if q[i] > q[prev_peakq]:
                    list_events[-1]["peakQ"] = i

                    if p[peakp] > p[prev_peakp]:
                        list_events[-1]["peakP"] = peakp
                continue

        list_events += [{"start": start, "end": end, "peakP": peakp, "peakQ": i}]

    return list_events


def _mask_event(
    instance: Model,
    peak_quant: float = 0.999,
    max_duration: int = 240,
    **unknown_options,
):

    _check_unknown_options_event_seg(unknown_options)

    mask = np.zeros(instance.input_data.qobs.shape)

    for i, catchment in enumerate(instance.mesh.code):

        prcp_tmp, qobs_tmp, ratio = _missing_values(
            instance.input_data.mean_prcp[i, :], instance.input_data.qobs[i, :]
        )

        if prcp_tmp is None:

            warnings.warn(
                f"Reject data at catchment {catchment} ({round(ratio * 100, 2)}% of missing values)"
            )

        else:

            list_events = _events_grad(prcp_tmp, qobs_tmp, peak_quant, max_duration)

            for event_number, t in enumerate(list_events):

                ts = t["start"]
                te = t["end"]

                mask[i, ts : te + 1] = event_number + 1

    return mask


def _event_segmentation(instance: Model, peak_quant: float, max_duration: int):

    date_range = pd.date_range(
        start=instance.setup.start_time,
        periods=instance.input_data.qobs.shape[1],
        freq=f"{int(instance.setup.dt)}s",
    )

    col_name = ["code", "start", "end", "maxrainfall", "flood", "season"]

    df = pd.DataFrame(columns=col_name)

    for i, catchment in enumerate(instance.mesh.code):

        prcp_tmp, qobs_tmp, ratio = _missing_values(
            instance.input_data.mean_prcp[i, :], instance.input_data.qobs[i, :]
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

            list_events = _events_grad(prcp_tmp, qobs_tmp, peak_quant, max_duration)

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


def _check_unknown_options_event_seg(unknown_options: dict):

    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        warnings.warn("Unknown event segmentation options: '%s'" % msg)


def _standardize_event_seg_options(options: dict | None) -> dict:

    if options is None:

        options = {}

    return options
