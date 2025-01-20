from __future__ import annotations

import warnings
from datetime import date, datetime
from typing import TYPE_CHECKING

import numpy as np

from smash._constant import MAX_DURATION, PEAK_QUANT
from smash.fcore._mwd_signatures import baseflow_separation as wrap_baseflow_separation

if TYPE_CHECKING:
    from smash.core.model.model import Model


def _get_season(now: datetime):
    year = 2000  # % dummy leap year to allow input X-02-29 (leap day)
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
    kpsh: bool = False,
):
    x = np.atleast_1d(x).astype("float64")
    if x.size < 3:
        return np.array([], dtype=int)

    # % find indices of all peaks
    dx = x[1:] - x[:-1]
    # % handle NaN's
    indnan = np.where(np.isnan(x))[0]

    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)

    ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]

    ind = np.unique(np.hstack((ine, ire, ife)))

    # % handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]

    # % first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]

    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]

    # % remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]

    # % remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])

    # % detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)

        for i in range(ind.size):
            if not idel[i]:
                # % keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) & (
                    x[ind[i]] > x[ind] if kpsh else True
                )
                idel[i] = 0  # % Keep current peak

        # % remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


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

    # % handle with missing values
    p = np.where(p < 0, np.nan, p)
    q = np.where(q < 0, np.nan, q)

    ind = _detect_peaks(q, mph=np.quantile(q[q > 0], peak_quant))
    list_events = []

    for i_peak in ind:
        p_search = p[range(max(i_peak - start_seg, 0), i_peak)]
        p_search_grad = np.gradient(p_search)

        ind_start = _detect_peaks(p_search_grad, mph=np.nanquantile(p_search_grad, rg_quant))

        if ind_start.size > 1:
            power = np.array(
                [
                    np.linalg.norm(
                        np.nan_to_num(p_search[j - 1 : j + st_power - 1], nan=0), ord=2
                    )  # remove power computation at nan values (replace nan by 0)
                    for j in ind_start
                ]
            )

            ind_start = ind_start[np.where(power > coef_re * max(power))[0]]

        elif ind_start.size == 0:
            ind_start = np.append(ind_start, np.nanargmax(p_search_grad))

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

        q_end = q[i_peak - 1 : fwindow - 1]
        mask = ~np.isnan(q_end)

        qbf = np.zeros(q_end.shape)

        bf = np.zeros(mask.sum(), dtype=np.float32)
        qf = bf.copy()

        wrap_baseflow_separation(q_end[mask], bf, qf, filter_parameter=0.925, passes=3)

        qbf[mask] = bf

        dflow = q_end - qbf
        dflow_windows = (dflow[i:] for i in range(min(end_search, dflow.size)))
        dflow = np.array([sum(i) for i in zip(*dflow_windows)])

        if np.isnan(dflow).all():  # if all nan values after the peak flow
            continue
        else:
            end = i_peak + np.nanargmin(dflow)

        if len(list_events) > 0:
            prev_start = list_events[-1]["start"]
            prev_end = list_events[-1]["end"]
            prev_peakq = list_events[-1]["peakQ"]
            prev_peakp = list_events[-1]["peakP"]

            # % detect double events:
            if prev_end >= start:
                list_events[-1]["end"] = max(end, prev_end)
                list_events[-1]["start"] = min(start, prev_start)
                list_events[-1]["multipeak"] = True

                if q[i_peak] > q[prev_peakq]:
                    list_events[-1]["peakQ"] = i_peak

                if p[peakp] > p[prev_peakp]:
                    list_events[-1]["peakP"] = peakp

                continue

        list_events.append({"start": start, "end": end, "multipeak": False, "peakP": peakp, "peakQ": i_peak})

    return list_events


def _mask_event(
    model: Model,
    peak_quant: float = PEAK_QUANT,
    max_duration: float = MAX_DURATION,  # in hour
) -> dict:
    mask = np.zeros(model.response_data.q.shape)
    n_event = np.zeros(model.mesh.ng)

    for i, catchment in enumerate(model.mesh.code):
        prcp = model.atmos_data.mean_prcp[i, :].copy()
        qobs = model.response_data.q[i, :].copy()

        if (prcp < 0).all() or (qobs < 0).all():
            warnings.warn(
                f"Catchment {catchment} has no observed precipitation or/and discharge data",
                stacklevel=2,
            )

        else:
            list_events = _events_grad(prcp, qobs, peak_quant, max_duration, model.setup.dt)

            n_event[i] = len(list_events)

            for event_number, t in enumerate(list_events):
                ts = t["start"]
                te = t["end"]

                mask[i, ts : te + 1] = event_number + 1

    return {"n": n_event, "mask": mask}
