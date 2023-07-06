from __future__ import annotations

from smash.signal_analysis.segmentation._tools import (
    _detect_peaks,
    _missing_values,
    _baseflow_separation,
    _get_season,
    _events_grad,
)

from smash.tools._common_function import _check_unknown_options

from smash._constant import PEAK_QUANT, MAX_DURATION

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

import numpy as np
import pandas as pd

import warnings


def _continuous_signatures(p: np.ndarray, q: np.ndarray, list_signatures: list[str]):
    res = []
    qb, qq = _baseflow_separation(q)
    qp = q[q >= 0]

    for signature in list_signatures:
        if signature == "Crc":
            try:
                res.append(np.sum(q) / np.sum(p))
            except:
                res.append(np.nan)

        elif signature == "Crchf":
            try:
                res.append(np.sum(qq) / np.sum(p))
            except:
                res.append(np.nan)

        elif signature == "Crclf":
            try:
                res.append(np.sum(qb) / np.sum(p))
            except:
                res.append(np.nan)

        elif signature == "Crch2r":
            try:
                res.append(np.sum(qq) / np.sum(q))
            except:
                res.append(np.nan)

        elif signature == "Cfp2":
            try:
                res.append(np.quantile(qp, 0.02))
            except:
                res.append(np.nan)

        elif signature == "Cfp10":
            try:
                res.append(np.quantile(qp, 0.1))
            except:
                res.append(np.nan)

        elif signature == "Cfp50":
            try:
                res.append(np.quantile(qp, 0.5))
            except:
                res.append(np.nan)

        elif signature == "Cfp90":
            try:
                res.append(np.quantile(qp, 0.9))
            except:
                res.append(np.nan)

    return res


def _event_signatures(
    p: np.ndarray,
    q: np.ndarray,
    start: int,
    peakp: float,
    peakq: float,
    list_signatures: list[str],
):
    res = []
    qb, qq = _baseflow_separation(q)
    deteq = True

    for signature in list_signatures:
        if signature == "Eff":
            res.append(np.mean(qq))

        elif signature == "Ebf":
            res.append(np.mean(qb))

        elif signature == "Erc":
            try:
                res.append(np.sum(q) / np.sum(p))
            except:
                res.append(np.nan)

        elif signature == "Erchf":
            try:
                res.append(np.sum(qq) / np.sum(p))
            except:
                res.append(np.nan)

        elif signature == "Erclf":
            try:
                res.append(np.sum(qb) / np.sum(p))
            except:
                res.append(np.nan)

        elif signature == "Erch2r":
            try:
                res.append(np.sum(qq) / np.sum(q))
            except:
                res.append(np.nan)

        elif signature == "Elt" or signature == "Epf":
            if deteq:
                deteq = False

                if peakq is None:
                    try:
                        peakq = (
                            _detect_peaks(q, mpd=len(q))[0] + start
                        )  # detect only 1 peak

                    except:
                        peakq = start + len(q) - 1

            if signature == "Elt":
                res.append(peakq - peakp)

            elif signature == "Epf":
                res.append(q[peakq - start])

    return res


def _sign_computation(
    instance: Model,
    cs: list[str],
    es: list[str],
    obs_comp: bool,
    sim_comp: bool,
    peak_quant: float = PEAK_QUANT,
    max_duration: float = MAX_DURATION,
    **unknown_options,
):
    if es:
        _check_unknown_options("event segmentation", unknown_options)

    prcp_cvt = (
        instance.atmos_data.mean_prcp
        * 0.001
        * instance.mesh.area_dln[..., np.newaxis]
        / instance.setup.dt
    )  # convert precip from mm to m3/s

    date_range = pd.date_range(
        start=instance.setup.start_time,
        periods=instance.obs_response.q.shape[1],
        freq=f"{int(instance.setup.dt)}s",
    )

    col_cs = ["code"] + cs
    col_es = ["code", "season", "start", "end"] + es

    dfsim_cs = pd.DataFrame(columns=col_cs)
    dfsim_es = pd.DataFrame(columns=col_es)

    dfobs_cs = pd.DataFrame(columns=col_cs)
    dfobs_es = pd.DataFrame(columns=col_es)

    if len(cs) + len(es) > 0:
        for i, catchment in enumerate(instance.mesh.code):
            prcp_tmp, qobs_tmp, ratio = _missing_values(
                prcp_cvt[i, :], instance.obs_response.q[i, :]
            )

            if prcp_tmp is None:
                warnings.warn(
                    f"Reject data at catchment {catchment} ({round(ratio * 100, 2)}% of missing values)"
                )

                row_cs = pd.DataFrame(
                    [[catchment] + [np.nan] * (len(col_cs) - 1)], columns=col_cs
                )
                row_es = pd.DataFrame(
                    [[catchment] + [np.nan] * (len(col_es) - 1)], columns=col_es
                )

                dfsim_cs = pd.concat([dfsim_cs, row_cs], ignore_index=True)
                dfsim_es = pd.concat([dfsim_es, row_es], ignore_index=True)

                dfobs_cs = pd.concat([dfobs_cs, row_cs], ignore_index=True)
                dfobs_es = pd.concat([dfobs_es, row_es], ignore_index=True)

            else:
                qsim_tmp = instance.sim_response.q[i, :].copy()

                if len(cs) > 0:
                    if sim_comp:
                        csignatures_sim = _continuous_signatures(
                            prcp_tmp, qsim_tmp, list_signatures=cs
                        )

                        rowsim_cs = pd.DataFrame(
                            [[catchment] + csignatures_sim], columns=col_cs
                        )

                        dfsim_cs = pd.concat([dfsim_cs, rowsim_cs], ignore_index=True)

                    if obs_comp:
                        csignatures_obs = _continuous_signatures(
                            prcp_tmp, qobs_tmp, list_signatures=cs
                        )

                        rowobs_cs = pd.DataFrame(
                            [[catchment] + csignatures_obs], columns=col_cs
                        )

                        dfobs_cs = pd.concat([dfobs_cs, rowobs_cs], ignore_index=True)

                if len(es) > 0:
                    list_events = _events_grad(
                        prcp_tmp, qobs_tmp, peak_quant, max_duration, instance.setup.dt
                    )

                    if len(list_events) == 0:
                        row_es = pd.DataFrame(
                            [[catchment] + [np.nan] * (len(col_es) - 1)],
                            columns=col_es,
                        )

                        dfsim_es = pd.concat([dfsim_es, row_es], ignore_index=True)
                        dfobs_es = pd.concat([dfobs_es, row_es], ignore_index=True)

                    else:
                        for t in list_events:
                            ts = t["start"]
                            te = t["end"]

                            event_prcp = prcp_tmp[ts : te + 1]
                            event_qobs = qobs_tmp[ts : te + 1]
                            event_qsim = qsim_tmp[ts : te + 1]

                            season = _get_season(date_range[ts].date())

                            if sim_comp:
                                esignatures_sim = _event_signatures(
                                    event_prcp,
                                    event_qsim,
                                    ts,
                                    t["peakP"],
                                    None,
                                    list_signatures=es,
                                )

                                rowsim_es = pd.DataFrame(
                                    [
                                        [
                                            catchment,
                                            season,
                                            date_range[ts],
                                            date_range[te],
                                        ]
                                        + esignatures_sim
                                    ],
                                    columns=col_es,
                                )

                                dfsim_es = pd.concat(
                                    [dfsim_es, rowsim_es], ignore_index=True
                                )

                            if obs_comp:
                                esignatures_obs = _event_signatures(
                                    event_prcp,
                                    event_qobs,
                                    ts,
                                    t["peakP"],
                                    t["peakQ"],
                                    list_signatures=es,
                                )

                                rowobs_es = pd.DataFrame(
                                    [
                                        [
                                            catchment,
                                            season,
                                            date_range[ts],
                                            date_range[te],
                                        ]
                                        + esignatures_obs
                                    ],
                                    columns=col_es,
                                )

                                dfobs_es = pd.concat(
                                    [dfobs_es, rowobs_es], ignore_index=True
                                )

    return dict(
        zip(
            ["cont", "event"],
            [
                {"obs": dfobs_cs, "sim": dfsim_cs},
                {"obs": dfobs_es, "sim": dfsim_es},
            ],
        )
    )
