from __future__ import annotations

from smash.solver._mw_forward import forward

from smash.core._constant import CSIGN, ESIGN

from smash.core._event_segmentation import (
    _detect_peaks,
    _missing_values,
    _events_grad,
    _baseflow_separation,
    _get_season,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from tqdm import tqdm
import warnings


def _standardize_signatures(sign: str | list[str] | None):

    if isinstance(sign, str):

        sign = [sign]

    elif sign is None:

        sign = CSIGN + ESIGN

    elif not isinstance(sign, list):
        raise ValueError(f"sign argument must be None, str or a list of str")

    for s in sign:
        if s not in CSIGN + ESIGN:
            raise ValueError(f"Unknown signature '{sign}'. Choices: {CSIGN + ESIGN}")

    cs = [s for s in sign if s[0] == "C"]
    es = [s for s in sign if s[0] == "E"]

    return cs, es


def _continuous_signatures(p: np.ndarray, q: np.ndarray, list_signatures: list[str]):

    res = []
    qb, qq = _baseflow_separation(q)
    qp = q[q > 0]

    for signature in list_signatures:

        if signature == "Crc":
            try:
                res += [np.sum(q) / np.sum(p)]
            except:
                res += [np.nan]

        if signature == "Crchf":
            try:
                res += [np.sum(qq) / np.sum(p)]
            except:
                res += [np.nan]

        if signature == "Crclf":
            try:
                res += [np.sum(qb) / np.sum(p)]
            except:
                res += [np.nan]

        if signature == "Crch2r":
            try:
                res += [np.sum(qq) / np.sum(q)]
            except:
                res += [np.nan]

        if signature == "Cfp2":
            try:
                res += [np.quantile(qp, 0.02)]
            except:
                res += [np.nan]

        if signature == "Cfp10":
            try:
                res += [np.quantile(qp, 0.1)]
            except:
                res += [np.nan]

        if signature == "Cfp50":
            try:
                res += [np.quantile(qp, 0.5)]
            except:
                res += [np.nan]

        if signature == "Cfp90":
            try:
                res += [np.quantile(qp, 0.9)]
            except:
                res += [np.nan]

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
            res += [np.mean(qq)]

        if signature == "Ebf":
            res += [np.mean(qb)]

        if signature == "Erc":
            try:
                res += [np.sum(q) / np.sum(p)]
            except:
                res += [np.nan]

        if signature == "Erchf":
            try:
                res += [np.sum(qq) / np.sum(p)]
            except:
                res += [np.nan]

        if signature == "Erclf":
            try:
                res += [np.sum(qb) / np.sum(p)]
            except:
                res += [np.nan]

        if signature == "Erch2r":
            try:
                res += [np.sum(qq) / np.sum(q)]
            except:
                res += [np.nan]

        if (signature == "Elt" or signature == "Epf") and deteq:

            deteq = False

            if peakq is None:

                try:
                    peakq = (
                        _detect_peaks(q, mpd=len(q))[0] + start
                    )  # detect only 1 peak

                except:
                    peakq = start + len(q) - 1

        if signature == "Elt":

            res += [peakq - peakp]

        if signature == "Epf":
            res += [q[peakq - start]]

    return res


def _signatures(
    instance: Model,
    cs: list[str],
    es: list[str],
    sign_obs: bool = True,
    warn: bool = True,
):

    prcp_cvt = (
        instance.input_data.mean_prcp.copy()
        * 0.001
        * instance.mesh.area[..., np.newaxis]
        / instance.setup.dt
    )  # convert precip from mm to m3/s

    date_range = pd.date_range(
        start=instance.setup.start_time,
        periods=instance.input_data.qobs.shape[1],
        freq=f"{int(instance.setup.dt)}s",
    )

    col_cs = ["code"] + [s + "_obs" for s in cs] * sign_obs + [s + "_sim" for s in cs]

    df_cs = pd.DataFrame(columns=col_cs)

    col_es = (
        ["code", "season", "start", "end"]
        + [s + "_obs" for s in es] * sign_obs
        + [s + "_sim" for s in es]
    )

    df_es = pd.DataFrame(columns=col_es)

    if len(cs) + len(es) > 0:

        for i, catchment in enumerate(instance.mesh.code):

            prcp_tmp, qobs_tmp, ratio = _missing_values(
                prcp_cvt[i, :], instance.input_data.qobs[i, :]
            )

            if prcp_tmp is None:

                if warn:
                    warnings.warn(
                        f"Reject data at catchment {catchment} ({round(ratio * 100, 2)}% of missing values)"
                    )

                row_cs = pd.DataFrame(
                    [[catchment] + [np.nan] * (len(col_cs) - 1)], columns=col_cs
                )

                df_cs = pd.concat([df_cs, row_cs], ignore_index=True)

                row_es = pd.DataFrame(
                    [[catchment] + [np.nan] * (len(col_es) - 1)], columns=col_es
                )

                df_es = pd.concat([df_es, row_es], ignore_index=True)

            else:

                qsim_tmp = instance.output.qsim[i, :].copy()

                if len(cs) > 0:

                    if sign_obs:

                        csignatures_obs = _continuous_signatures(
                            prcp_tmp, qobs_tmp, list_signatures=cs
                        )

                    else:
                        csignatures_obs = []

                    csignatures_sim = _continuous_signatures(
                        prcp_tmp, qsim_tmp, list_signatures=cs
                    )

                    row_cs = pd.DataFrame(
                        [[catchment] + csignatures_obs + csignatures_sim],
                        columns=col_cs,
                    )

                    df_cs = pd.concat([df_cs, row_cs], ignore_index=True)

                if len(es) > 0:

                    list_events = _events_grad(prcp_tmp, qobs_tmp)

                    if len(list_events) == 0:

                        row_es = pd.DataFrame(
                            [[catchment] + [np.nan] * (len(col_es) - 1)],
                            columns=col_es,
                        )

                        df_es = pd.concat([df_es, row_es], ignore_index=True)

                    else:

                        for t in list_events:

                            ts = t["start"]
                            te = t["end"]

                            event_prcp = prcp_tmp[ts:te]
                            event_qobs = qobs_tmp[ts:te]
                            event_qsim = qsim_tmp[ts:te]

                            season = _get_season(date_range[ts].date())

                            if sign_obs:
                                esignatures_obs = _event_signatures(
                                    event_prcp,
                                    event_qobs,
                                    ts,
                                    t["peakP"],
                                    t["peakQ"],
                                    list_signatures=es,
                                )
                            else:
                                esignatures_obs = []

                            esignatures_sim = _event_signatures(
                                event_prcp,
                                event_qsim,
                                ts,
                                t["peakP"],
                                None,
                                list_signatures=es,
                            )

                            row_es = pd.DataFrame(
                                [
                                    [catchment, season, date_range[ts], date_range[te]]
                                    + esignatures_obs
                                    + esignatures_sim
                                ],
                                columns=col_es,
                            )

                            df_es = pd.concat([df_es, row_es], ignore_index=True)

    return {"C": df_cs, "E": df_es}


def _sa_todf(problem: dict, y: np.ndarray, yname: list[str]):

    si = sobol.analyze(problem, y)
    total_si, first_si, second_si = si.to_df()
    st = [f"{yname}.ST_{factor}" for factor in total_si.index]
    s1 = [f"{yname}.S1_{factor}" for factor in first_si.index]
    s2 = [f"{yname}.S2_{factor[0]}-{factor[1]}" for factor in second_si.index]
    df = pd.DataFrame(columns=st + s1 + s2)
    arr = np.concatenate(
        (
            total_si.iloc[:, 0].to_numpy(),
            first_si.iloc[:, 0].to_numpy(),
            second_si.iloc[:, 0].to_numpy(),
        ),
        axis=None,
    )
    df.loc[len(df)] = arr

    return df


def _signatures_sensitivity(
    instance: Model, problem: dict, sample: pd.DataFrame, cs: list[str], es: list[str]
):

    df_cs = []
    df_es = []

    for i in tqdm(range(len(sample)), desc="Computing signatures sensitivity"):

        for k in sample.keys():
            setattr(instance.parameters, k, sample[k][i])

        cost = np.float32(0)

        forward(
            instance.setup,
            instance.mesh,
            instance.input_data,
            instance.parameters,
            instance.parameters.copy(),
            instance.states,
            instance.states.copy(),
            instance.output,
            cost,
        )

        res_sign = _signatures(instance, cs, es, sign_obs=False, warn=(i == 0))

        df_cs += [res_sign["C"]]
        df_es += [res_sign["E"]]

    dict_df = {"C": df_cs, "E": df_es}

    df = {
        "C": dict_df["C"][0][["code"]],
        "E": dict_df["E"][0][["code", "season", "start", "end"]],
    }

    dict_sname = {}

    dict_sname["C"] = [
        name.split("_")[0] for name in list(dict_df["C"][0]) if name.startswith("C")
    ]

    dict_sname["E"] = [
        name.split("_")[0] for name in list(dict_df["E"][0]) if name.startswith("E")
    ]

    for df_mc, sname, stype in zip(
        dict_df.values(), dict_sname.values(), list(dict_sname.keys())
    ):

        dfsa_col = []

        for k in [
            s for s in df_mc[0].keys() if s.startswith(tuple(name for name in sname))
        ]:

            dfsa = []

            for j in range(len(df_mc[0])):

                y = np.array([df_mc[i][k].loc[j] for i in range(len(df_mc))])

                dfsa += [_sa_todf(problem, y, k)]

            dfsa_col += [pd.concat(dfsa, ignore_index=True)]

        df[stype] = pd.concat([df[stype]] + dfsa_col, axis=1, join="inner")

    return df
