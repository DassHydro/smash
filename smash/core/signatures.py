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
from SALib.sample.sobol import sample as sb_generate_sample
from SALib.analyze.sobol import analyze as sb_analyze
from tqdm import tqdm
import warnings


class SignResult(dict):
    """
    Represents signatures computation result.

    Notes
    -----
    This class is essentially a subclass of dict with attribute accessors.

    Attributes
    ----------
    cont : dict
        Continuous signatures. The keys are

        - 'obs': a dataframe representing observation results.
        - 'sim': a dataframe representing simulation results.

        The column names of both dataframes consist of the catchment code and studied signature names.

    event : dict
        Flood event signatures. The keys are

        - 'obs': a dataframe representing observation results.
        - 'sim': a dataframe representing simulation results.

        The column names of both dataframes consist of the catchment code, the season that event occurrs, the beginning/end of each event and studied signature names.

    See Also
    --------
    Model.signatures: Compute continuous or/and flood event signatures of the Model.

    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return "\n".join(
                [k.rjust(m) + ": " + repr(v) for k, v in sorted(self.items())]
            )
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class SignSensResult(dict):
    """
    Represents signatures sensitivity computation result.

    Notes
    -----
    This class is essentially a subclass of dict with attribute accessors.

    Attributes
    ----------
    cont : dict
        A dictionary with two keys

        - 'total_si' : representing the total-order Sobol indices of the hydrological model parameters on continuous signatures.
        - 'first_si' : representing the first-order Sobol indices of the hydrological model parameters on continuous signatures.

        Each value of the dictionary is a sub-dictionary with the keys are the hydrological model parameters.
        Then each value of each sub-dictionary (associating to a model parameter) is a dataframe containing the sensitivity computation results of the associated model paramter on all studied signatures.
        The column names of each dataframe consist of the catchment code and studied signature names.

    event : dict
        A dictionary with two keys

        - 'total_si' : representing the total-order Sobol indices of the hydrological model parameters on flood event signatures.
        - 'first_si' : representing the first-order Sobol indices of the hydrological model parameters on flood event signatures.

        Each value of the dictionary is a sub-dictionary with the keys are the hydrological model parameters.
        Then each value of each sub-dictionary (associating to a model parameter) is a dataframe containing the sensitivity computation results of the associated model paramter on all studied signatures.
        The column names of each dataframe consist of the catchment code, the season that event occurrs, the beginning/end of each event and studied signature names.

    sample: pandas.DataFrame
        A dataframe containing the generated samples used to compute sensitivity indices.

    See Also
    --------
    Model.signatures_sensitivity: Compute the first- and total-order variance-based sensitivity (Sobol indices) of spatially uniform hydrological model parameters on the output signatures.

    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return "\n".join(
                [k.rjust(m) + ": " + repr(v) for k, v in sorted(self.items())]
            )
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


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
                res.append(np.sum(q) / np.sum(p))
            except:
                res.append(np.nan)

        if signature == "Crchf":
            try:
                res.append(np.sum(qq) / np.sum(p))
            except:
                res.append(np.nan)

        if signature == "Crclf":
            try:
                res.append(np.sum(qb) / np.sum(p))
            except:
                res.append(np.nan)

        if signature == "Crch2r":
            try:
                res.append(np.sum(qq) / np.sum(q))
            except:
                res.append(np.nan)

        if signature == "Cfp2":
            try:
                res.append(np.quantile(qp, 0.02))
            except:
                res.append(np.nan)

        if signature == "Cfp10":
            try:
                res.append(np.quantile(qp, 0.1))
            except:
                res.append(np.nan)

        if signature == "Cfp50":
            try:
                res.append(np.quantile(qp, 0.5))
            except:
                res.append(np.nan)

        if signature == "Cfp90":
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

        if signature == "Ebf":
            res.append(np.mean(qb))

        if signature == "Erc":
            try:
                res.append(np.sum(q) / np.sum(p))
            except:
                res.append(np.nan)

        if signature == "Erchf":
            try:
                res.append(np.sum(qq) / np.sum(p))
            except:
                res.append(np.nan)

        if signature == "Erclf":
            try:
                res.append(np.sum(qb) / np.sum(p))
            except:
                res.append(np.nan)

        if signature == "Erch2r":
            try:
                res.append(np.sum(qq) / np.sum(q))
            except:
                res.append(np.nan)

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

            res.append(peakq - peakp)

        if signature == "Epf":
            res.append(q[peakq - start])

    return res


def _signatures(
    instance: Model,
    cs: list[str],
    es: list[str],
    obs_comp: bool = True,  # decide if process observation computation or not.
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

    col_cs = ["code"] + cs
    col_es = ["code", "season", "start", "end"] + es

    dfsim_cs = pd.DataFrame(columns=col_cs)
    dfsim_es = pd.DataFrame(columns=col_es)

    dfobs_cs = pd.DataFrame(columns=col_cs)
    dfobs_es = pd.DataFrame(columns=col_es)

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
                row_es = pd.DataFrame(
                    [[catchment] + [np.nan] * (len(col_es) - 1)], columns=col_es
                )

                dfsim_cs = pd.concat([dfsim_cs, row_cs], ignore_index=True)
                dfsim_es = pd.concat([dfsim_es, row_es], ignore_index=True)

                dfobs_cs = pd.concat([dfobs_cs, row_cs], ignore_index=True)
                dfobs_es = pd.concat([dfobs_es, row_es], ignore_index=True)

            else:

                qsim_tmp = instance.output.qsim[i, :].copy()

                if len(cs) > 0:

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

                    list_events = _events_grad(prcp_tmp, qobs_tmp)

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

                            event_prcp = prcp_tmp[ts:te]
                            event_qobs = qobs_tmp[ts:te]
                            event_qsim = qsim_tmp[ts:te]

                            season = _get_season(date_range[ts].date())

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
                                    [catchment, season, date_range[ts], date_range[te]]
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

    return SignResult(
        dict(
            zip(
                ["cont", "event"],
                [
                    {"obs": dfobs_cs, "sim": dfsim_cs},
                    {"obs": dfobs_es, "sim": dfsim_es},
                ],
            )
        )
    )


def _signatures_sensitivity(
    instance: Model,
    problem: dict,
    n: int,
    cs: list[str],
    es: list[str],
    seed: None | int,
):

    # generate samples
    sample = sb_generate_sample(problem, n, calc_second_order=False, seed=seed)

    # signatures computation
    dfs_cs = []  # list of dataframes concerned to CS
    dfs_es = []  # list of dataframes concerned to ES

    for i in tqdm(range(len(sample)), desc="</> Computing signatures sensitivity"):

        for j, name in enumerate(problem["names"]):
            setattr(instance.parameters, name, sample[i, j])

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

        res_sign = _signatures(instance, cs, es, obs_comp=False, warn=(i == 0))

        dfs_cs.append(res_sign.cont["sim"])
        dfs_es.append(res_sign.event["sim"])

    # sensitivity computation
    dfinfo_cs = dfs_cs[0][["code"]]
    dfinfo_es = dfs_es[0][["code", "season", "start", "end"]]

    res_sa = []

    for dfinfo, dfs, signs in zip(
        [dfinfo_cs, dfinfo_es], [dfs_cs, dfs_es], [cs, es]
    ):  # loop for CS and ES

        dict_sa_tot = {key: [] for key in problem["names"]}
        dict_sa_first = {key: [] for key in problem["names"]}

        for name in signs:

            total_si = {key: [] for key in problem["names"]}
            first_si = {key: [] for key in problem["names"]}

            for j in range(len(dfinfo["code"])):

                y = np.array([dfs[i][name].loc[j] for i in range(len(dfs))])

                tsi, fsi = sb_analyze(problem, y, calc_second_order=False).to_df()

                for ip, param in enumerate(problem["names"]):

                    total_si[param].append(tsi.iloc[:, 0].iloc[ip])

                    first_si[param].append(fsi.iloc[:, 0].iloc[ip])

            for param in problem["names"]:

                dict_sa_tot[param].append(pd.Series(total_si[param], name=name))

                dict_sa_first[param].append(pd.Series(first_si[param], name=name))

        res_sa.append(
            {
                "total_si": {
                    param: pd.concat(
                        [dfinfo] + dict_sa_tot[param], axis=1, join="inner"
                    )
                    for param in problem["names"]
                },
                "first_si": {
                    param: pd.concat(
                        [dfinfo] + dict_sa_first[param], axis=1, join="inner"
                    )
                    for param in problem["names"]
                },
            }
        )  # concat dfinfo and dict_sa

    return SignSensResult(
        dict(
            zip(
                ["cont", "event", "sample"],
                res_sa + [pd.DataFrame(sample, columns=problem["names"])],
            )
        )
    )
