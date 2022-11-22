from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date, datetime

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model


def _missing_values(p, q, keep=0.2, method="nearest"):
    """Deal with misssing values problem on P and Q"""

    df = pd.DataFrame({"P": p, "Q": q})
    df.P = np.where(df.P < 0, np.nan, df.P)
    df.Q = np.where(df.Q < 0, np.nan, df.Q)
    MVrat = max(df.P.isna().sum() / len(p), df.Q.isna().sum() / len(q))
    if MVrat <= keep:
        indP = np.where(df.P.to_numpy() >= 0)
        indQ = np.where(df.Q.to_numpy() >= 0)
        df.P[0] = df.P[indP[0][0]]
        df.P[len(df) - 1] = df.P[indP[0][-1]]
        df.Q[0] = df.Q[indQ[0][0]]
        df.Q[len(df) - 1] = df.Q[indQ[0][-1]]
        df.interpolate(method=method, inplace=True)
        return df.P.to_numpy(), df.Q.to_numpy(), MVrat
    else:
        return None, None, MVrat


def _BaseflowSeparation(streamflow, filter_parameter=0.925, passes=3):

    n = len(streamflow)
    Ends = np.array([n - 1 if i % 2 == 1 else 0 for i in range(passes + 1)])
    AddToStart = np.array([-1 if i % 2 == 1 else 1 for i in range(passes)])

    btP = np.copy(streamflow)  # Previous pass's baseflow approximation
    qft = np.zeros(n)
    bt = np.zeros(n)
    if streamflow[0] < np.quantile(streamflow, 0.25):
        bt[0] = streamflow[0]
    else:
        bt[0] = np.mean(streamflow) / 1.5
    ##Guess baseflow value in first time step.
    for j in range(passes):
        rang = np.linspace(
            Ends[j] + AddToStart[j],
            Ends[j + 1],
            np.abs(Ends[j] + AddToStart[j] - Ends[j + 1]) + 1,
        )
        for ii in rang:
            i = int(ii)
            if (
                filter_parameter * bt[i - AddToStart[j]]
                + ((1 - filter_parameter) / 2) * (btP[i] + btP[i - AddToStart[j]])
                > btP[i]
            ):
                bt[i] = btP[i]
            else:
                bt[i] = filter_parameter * bt[i - AddToStart[j]] + (
                    (1 - filter_parameter) / 2
                ) * (btP[i] + btP[i - AddToStart[j]])
            qft[i] = streamflow[i] - bt[i]
        if j < passes - 1:
            btP = np.copy(bt)
            if streamflow[Ends[j + 1]] < np.mean(btP):
                bt[Ends[j + 1]] = streamflow[Ends[j + 1]] / 1.2
            else:
                bt[Ends[j + 1]] = np.mean(btP)
    return bt, qft


def _get_season(now):

    Y = 2000  # dummy leap year to allow input X-02-29 (leap day)
    seasons = [
        ("winter", (date(Y, 1, 1), date(Y, 3, 20))),
        ("spring", (date(Y, 3, 21), date(Y, 6, 20))),
        ("summer", (date(Y, 6, 21), date(Y, 9, 22))),
        ("autumn", (date(Y, 9, 23), date(Y, 12, 20))),
        ("winter", (date(Y, 12, 21), date(Y, 12, 31))),
    ]
    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(season for season, (start, end) in seasons if start <= now <= end)


def _detect_peaks(
    x,
    mph=None,
    mpd=1,
    threshold=0,
    edge="rising",
    kpsh=False,
    valley=False,
):

    x = np.atleast_1d(x).astype("float64")
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
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
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[
            np.in1d(
                ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True
            )
        ]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) & (
                    x[ind[i]] > x[ind] if kpsh else True
                )
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def _events_grad(
    p,
    q,
    thresgrad=0.8,
    rpower=0.2,
    dh_search=72,
    max_duration=10 * 24,
    thres_quant=0.999,
    search_start=12,
    search_end=24,
):

    ind = _detect_peaks(q, mph=np.quantile(q[q > 0], thres_quant))
    list_events = []
    for i in ind:
        P_search = p[range(max(i - dh_search, 0), i)]
        P_search_grad = np.gradient(P_search)
        try:
            ind_start = _detect_peaks(
                P_search_grad, mph=np.quantile(P_search_grad, thresgrad)
            )
        except:
            continue
        power = np.array(
            [
                np.linalg.norm(P_search[j - 1 : j + search_start - 1], ord=2)
                for j in ind_start
            ]
        )
        try:
            ind_start = ind_start[np.where(power > rpower * max(power))[0]]
        except:
            continue
        try:
            ind_start_minQ = ind_start[0]
        except:
            continue
        start = ind_start_minQ + max(i - dh_search, 0)
        try:
            peakP = _detect_peaks(p[start:i], mpd=len(p))[0]
        except:
            continue
        peakP += start
        qbf = _BaseflowSeparation(q[i - 1 : start + max_duration + search_end - 1])[0]
        dflow = q[i - 1 : start + max_duration + search_end - 1] - qbf
        dflow = np.array([sum(i) for i in zip(*(dflow[i:] for i in range(search_end)))])
        end = i + np.argmin(dflow)
        if len(list_events) > 0:
            prev_start = list_events[-1]["start"]
            prev_end = list_events[-1]["end"]
            prev_peakQ = list_events[-1]["peakQ"]
            prev_peakP = list_events[-1]["peakP"]
            # merge two events respecting to max duration:
            if max(end, prev_end) <= prev_start + max_duration:
                list_events[-1]["end"] = max(end, prev_end)
                if q[i] > q[prev_peakQ]:
                    list_events[-1]["peakQ"] = i
                    if p[peakP] > p[prev_peakP]:
                        list_events[-1]["peakP"] = peakP
                continue
        list_events += [{"start": start, "end": end, "peakP": peakP, "peakQ": i}]
    return list_events


def _mask_event(instance: Model, season="all"):

    po = instance.input_data.mean_prcp 
    qo = instance.input_data.qobs


    first_ts = instance.setup.start_time
    dtserie = pd.date_range(
        start=first_ts, periods=qo.shape[1], freq=f"{int(instance.setup.dt)}s"
    )

    mask = np.zeros(qo.shape)

    pobs = []
    qobs = []

    list_events_all = []

    for i, catchment in enumerate(instance.mesh.code):

        pobs_tmp, qobs_tmp, ratio = _missing_values(po[i, :], qo[i, :])

        if pobs_tmp is None:

            print(
                f"Reject data at catchment {catchment} ({round(ratio*100,2)}% of missing values)"
            )

        else:

            pobs += [pobs_tmp]  # in mm for plotting
            qobs += [qobs_tmp]  # in m3/s for plotting

            list_events = _events_grad(pobs_tmp, qobs_tmp)
            list_events_all += [list_events]

            for event_number,t in enumerate(list_events):
                ts = t["start"]
                te = t["end"]
                if season == "all":
                    mask[i, ts : te + 1] = event_number + 1
                elif season in ["spring", "summer", "autumn", "winter"]:
                    if season == _get_season(dtserie[ts].date()):
                        mask[i, ts : te + 1] = event_number + 1
                    else:
                        pass
                else:
                    raise KeyError(
                        f'season must be "spring", "summer", "autumn", "winter" or "all" !'
                    )

    pobs = np.array(pobs)
    qobs = np.array(qobs)

    return (pobs, qobs, list_events_all, dtserie), mask

def _date_segmentation(instance: Model):

    po = instance.input_data.mean_prcp 
    qo = instance.input_data.qobs


    first_ts = instance.setup.start_time
    dtserie = pd.date_range(
        start=first_ts, periods=qo.shape[1], freq=f"{int(instance.setup.dt)}s"
    )

    col_name = ['catchment', 'start', 'end', 'maxrainfall', 'flood', 'season']
    df = pd.DataFrame(columns=col_name)

    for i, catchment in enumerate(instance.mesh.code):

        pobs_tmp, qobs_tmp, ratio = _missing_values(po[i, :], qo[i, :])

        if pobs_tmp is None:

            print(
                f"Reject data at catchment {catchment} ({round(ratio*100,2)}% of missing values)"
            )

            pdrow = pd.DataFrame(
                [[catchment] + [np.nan] * (len(col_name) - 1)], columns=col_name
            )
            df = pd.concat([df, pdrow], ignore_index=True)

        else:

            list_events = _events_grad(pobs_tmp, qobs_tmp)

            for t in list_events:
                ts = dtserie[t["start"]]
                te = dtserie[t["end"]]
                peakQ = dtserie[t['peakQ']]
                peakP = dtserie[t['peakP']]
                season = _get_season(ts)

                pdrow = pd.DataFrame(
                    [[catchment, ts, te, peakP, peakQ, season]], columns=col_name
                )
                df = pd.concat([df, pdrow], ignore_index=True)

    return df
