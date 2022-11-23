from __future__ import annotations

import numpy as np
import pandas as pd
from SALib.analyze import sobol

from smash.core._event_segmentation import (
    _detect_peaks,
    _missing_values,
    _events_grad,
    _BaseflowSeparation,
    _get_season,
)

__all__ = ["signatures", "signatures_sensitivity"]


CSIGN = ["Crc", "Crchf", "Crclf", "Crch2r", "Cfp2", "Cfp10", "Cfp50", "Cfp90"]
ESIGN = ["Eff", "Ebf", "Erc", "Erchf", "Erclf", "Erch2r", "Elt", "Epf"]

def _standardize_signatures(S):

    if isinstance(S, str):
        S = [S]
    elif S == None:
        S = CSIGN + ESIGN
    elif not isinstance(S, list):
        raise ValueError(
            f"Signature(s) should be None or str or a list of str (not be {S})"
        )

    if not all([s in CSIGN + ESIGN for s in S]):
        raise KeyError(f"At least one of signatures in {S} is not represented in list {CSIGN + ESIGN}")

    CS = [s for s in S if s[0]=="C"]
    ES = [s for s in S if s[0]=="E"]

    return CS, ES

def _signature_continu(P, Q, list_signatures):

    res = []
    Qb, Qq = _BaseflowSeparation(Q)
    Qp = Q[Q > 0]

    for signature in list_signatures:

        if signature == "Crc":
            try:
                res += [np.sum(Q) / np.sum(P)]
            except:
                res += [np.nan]

        if signature == "Crchf":
            try:
                res += [np.sum(Qq) / np.sum(P)]
            except:
                res += [np.nan]

        if signature == "Crclf":
            try:
                res += [np.sum(Qb) / np.sum(P)]
            except:
                res += [np.nan]

        if signature == "Crch2r":
            try:
                res += [np.sum(Qq) / np.sum(Q)]
            except:
                res += [np.nan]

        if signature == "Cfp2":
            try:
                res += [np.quantile(Qp, 0.02)]
            except:
                res += [np.nan]

        if signature == "Cfp10":
            try:
                res += [np.quantile(Qp, 0.1)]
            except:
                res += [np.nan]

        if signature == "Cfp50":
            try:
                res += [np.quantile(Qp, 0.5)]
            except:
                res += [np.nan]

        if signature == "Cfp90":
            try:
                res += [np.quantile(Qp, 0.9)]
            except:
                res += [np.nan]

    return res

def _signature_event(P, Q, start, peakP, peakQ, list_signatures):

    res = []
    Qb, Qq = _BaseflowSeparation(Q)
    deteQ = True

    for signature in list_signatures:

        if signature == "Eff":
            res += [np.mean(Qq)]

        if signature == "Ebf":
            res += [np.mean(Qb)]

        if signature == "Erc":
            try:
                res += [np.sum(Q) / np.sum(P)]
            except:
                res += [np.nan]

        if signature == "Erchf":
            try:
                res += [np.sum(Qq) / np.sum(P)]
            except:
                res += [np.nan]

        if signature == "Erclf":
            try:
                res += [np.sum(Qb) / np.sum(P)]
            except:
                res += [np.nan]

        if signature == "Erch2r":
            try:
                res += [np.sum(Qq) / np.sum(Q)]
            except:
                res += [np.nan]

        if (signature == "Elt" or signature == "Epf") and deteQ:
            deteQ = False
            if peakQ is None:
                try:
                    peakQ = (
                        _detect_peaks(Q, mpd=len(Q))[0] + start
                    )  # detect only 1 peak
                except:
                    peakQ = start + len(Q) - 1

        if signature == "Elt":
            res += [peakQ - peakP]

        if signature == "Epf":
            res += [Q[peakQ - start]]

    return res

def _signatures_one_basin(
    instance, Cname, Ename, sign_obs=True
):

    po = instance.input_data.mean_prcp
    po = po * 0.001 * instance.mesh.area[..., np.newaxis] / instance.setup.dt # convert precip from mm to m3/s
    qo = instance.input_data.qobs
    qs = instance.output.qsim

    list_catch = instance.mesh.code

    first_ts = instance.setup.start_time
    dtserie = pd.date_range(
        start=first_ts, periods=qo.shape[1], freq=f"{int(instance.setup.dt)}s"
    )

    col_nameC = (
        ["catchment"]
        + [s + "_obs" for s in Cname] * sign_obs
        + [s + "_sim" for s in Cname]
    )
    dfC = pd.DataFrame(columns=col_nameC)

    col_nameE = (
        ["catchment", "season", "start", "end"]
        + [s + "_obs" for s in Ename] * sign_obs
        + [s + "_sim" for s in Ename]
    )
    dfE = pd.DataFrame(columns=col_nameE)

    if len(Ename)+len(Cname) > 0:
        
        for i, catchment in enumerate(list_catch):

            pobs_tmp, qobs_tmp, ratio = _missing_values(po[i, :], qo[i, :])

            if pobs_tmp is None:

                print(
                    f"Reject data at catchment {catchment} ({round(ratio*100,2)}% of missing values)"
                )

                pdrowC = pd.DataFrame(
                    [[catchment] + [np.nan] * (len(col_nameC) - 1)], columns=col_nameC
                )
                dfC = pd.concat([dfC, pdrowC], ignore_index=True)

                pdrowE = pd.DataFrame(
                    [[catchment] + [np.nan] * (len(col_nameE) - 1)], columns=col_nameE
                )
                dfE = pd.concat([dfE, pdrowE], ignore_index=True)

            else:

                qcal_tmp = np.copy(qs[i, :])
                qcal_tmp = qcal_tmp[~np.isnan(qcal_tmp)]

                if len(Cname) > 0:

                    if sign_obs:

                        CSignaturesObs = _signature_continu(
                            pobs_tmp, qobs_tmp, list_signatures=Cname
                        )

                    else:
                        CSignaturesObs = []

                    CSignaturesCal = _signature_continu(
                        pobs_tmp, qcal_tmp, list_signatures=Cname
                    )

                    pdrowC = pd.DataFrame(
                        [[catchment] + CSignaturesObs + CSignaturesCal], columns=col_nameC
                    )
                    dfC = pd.concat([dfC, pdrowC], ignore_index=True)

                if len(Ename) > 0:

                    list_events = _events_grad(pobs_tmp, qobs_tmp)

                    if len(list_events) == 0:

                        pdrowE = pd.DataFrame(
                            [[catchment] + [np.nan] * (len(col_nameE) - 1)],
                            columns=col_nameE,
                        )
                        dfE = pd.concat([dfE, pdrowE], ignore_index=True)

                    else:

                        for t in list_events:

                            ts = t["start"]
                            te = t["end"]

                            pobs_ = pobs_tmp[ts:te]
                            qcal_ = qcal_tmp[ts:te]
                            qobs_ = qobs_tmp[ts:te]

                            season = _get_season(dtserie[ts].date())

                            if sign_obs:
                                ESignaturesObs = _signature_event(
                                    pobs_,
                                    qobs_,
                                    ts,
                                    t["peakP"],
                                    t["peakQ"],
                                    list_signatures=Ename,
                                )
                            else:
                                ESignaturesObs = []

                            ESignaturesCal = _signature_event(
                                pobs_, qcal_, ts, t["peakP"], None, list_signatures=Ename
                            )

                            pdrowE = pd.DataFrame(
                                [
                                    [catchment, season, dtserie[ts], dtserie[te]]
                                    + ESignaturesObs
                                    + ESignaturesCal
                                ],
                                columns=col_nameE,
                            )
                            dfE = pd.concat([dfE, pdrowE], ignore_index=True)

    return {"C": dfC, "E": dfE}, {"C": Cname, "E": Ename}

def _SA_todf(problem, Y, Yname):

    Si = sobol.analyze(problem, Y)
    total_Si, first_Si, second_Si = Si.to_df()
    ST = [f"{Yname}.ST_{factor}" for factor in total_Si.index]
    S1 = [f"{Yname}.S1_{factor}" for factor in first_Si.index]
    S2 = [f"{Yname}.S2_{factor[0]}-{factor[1]}" for factor in second_Si.index]
    df = pd.DataFrame(columns=ST + S1 + S2)
    arr = np.concatenate(
        (
            total_Si.iloc[:, 0].to_numpy(),
            first_Si.iloc[:, 0].to_numpy(),
            second_Si.iloc[:, 0].to_numpy(),
        ),
        axis=None,
    )
    df.loc[len(df)] = arr

    return df

def _signatures_sensitivity_one_basin(problem, dict_df, dict_Sname):

    df = {
        "C": dict_df["C"][0][["catchment"]],
        "E": dict_df["E"][0][["catchment", "season", "start", "end"]],
    }

    for df_MC, Sname, Stype in zip(
        dict_df.values(), dict_Sname.values(), list(dict_Sname.keys())
    ):

        dfSA_col = []

        for k in [
            s for s in df_MC[0].keys() if s.startswith(tuple(name for name in Sname))
        ]:

            dfSA = []

            for j in range(len(df_MC[0])):

                Y = np.array([df_MC[i][k].loc[j] for i in range(len(df_MC))])

                dfSA += [_SA_todf(problem, Y, k)]

            dfSA_col += [pd.concat(dfSA, ignore_index=True)]

        df[Stype] = pd.concat([df[Stype]] + dfSA_col, axis=1, join="inner")

    return df
