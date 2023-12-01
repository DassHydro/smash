from __future__ import annotations

from smash._constant import PEAK_QUANT, MAX_DURATION

from smash.fcore._mwd_signatures import (
    rc as wrap_rc,
    rchf as wrap_rchf,
    rclf as wrap_rclf,
    rch2r as wrap_rch2r,
    cfp as wrap_cfp,
    eff as wrap_eff,
    ebf as wrap_ebf,
    epf as wrap_epf,
    elt as wrap_elt,
)

from smash.core.signal_analysis.signatures._standardize import (
    _standardize_signatures_args,
)

from smash.core.signal_analysis.segmentation._tools import _events_grad, _get_season

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.util._typing import ListLike

import numpy as np
import pandas as pd
import warnings


__all__ = ["Signatures", "signatures"]


class Signatures:
    """
    Represents signatures computation result.

    Attributes
    ----------
    cont : dict
        A dataframe representing observed or simulated continuous signatures.
        The column names consist of the catchment code and studied signature names.

    event : dict
        A dataframe representing observed or simulated flood event signatures.
        The column names consist of the catchment code, the season that event occurrs, the beginning/end of each event and studied signature names.

    See Also
    --------
    smash.signatures : Compute simulated or observed hydrological signatures of the Model.

    """

    def __init__(self, data: dict | None = None):
        if data is None:
            data = {}

        self.__dict__.update(data)

    def __repr__(self):
        dct = self.__dict__

        if dct.keys():
            m = max(map(len, list(dct.keys()))) + 1
            return "\n".join(
                [
                    k.rjust(m) + ": " + repr(type(v))
                    for k, v in sorted(dct.items())
                    if not k.startswith("_")
                ]
            )
        else:
            return self.__class__.__name__ + "()"


def signatures(
    model: Model,
    sign: str | ListLike | None = None,
    event_seg: dict | None = None,
    domain: str = "obs",
):
    """
    Compute simulated or observed hydrological signatures of the Model.

    .. hint::
        See the (TODO: Fill) for more.

    Parameters
    ----------
    model : Model
        Model object.

    sign : str, list of str or None, default None
        Define signature(s) to compute. Should be one of

        - 'Crc', 'Crchf', 'Crclf', 'Crch2r', 'Cfp2', 'Cfp10', 'Cfp50', 'Cfp90' (continuous signatures)
        - 'Eff', 'Ebf', 'Erc', 'Erchf', 'Erclf', 'Erch2r', 'Elt', 'Epf' (flood event signatures)

        .. note::
            If not given, all of continuous and flood event signatures will be computed.

    event_seg : dict or None, default None
        A dictionary of event segmentation options when calculating flood event signatures. The keys are 'peak_quant', 'max_duration', and 'by'.

        .. note::
            If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element. See `smash.hydrograph_segmentation` for more.

    domain : str, default 'obs'
        Compute observed (obs) or simulated (sim) signatures.

    Returns
    -------
    res : Signatures
        The signatures computation results represented as a `Signatures` object.

    See Also
    --------
    Signatures : Represents signatures computation result.

    Examples
    --------
    >>> import smash
    >>> from smash.factory import load_dataset
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Run the forward Model:

    >>> model.forward_run()

    Compute simulated signatures:

    >>> sign = smash.signatures(model, domain="sim")
    >>> sign
     cont: <class 'pandas.core.frame.DataFrame'>
    event: <class 'pandas.core.frame.DataFrame'>

    >>> sign.event  # simulated flood event signatures
           code  season               start  ...    Erch2r  Elt        Epf
    0  V3524010  autumn 2014-11-03 03:00:00  ...  0.214772  3.0  85.736832
    1  V3515010  autumn 2014-11-03 10:00:00  ...  0.202139  0.0  17.256138
    2  V3517010  autumn 2014-11-03 08:00:00  ...  0.187440  1.0   4.770674

    [3 rows x 12 columns]
    """

    cs, es, domain, event_seg = _standardize_signatures_args(sign, domain, event_seg)

    res = _signatures(model, cs, es, domain, **event_seg)

    return Signatures(res)


# TODO: Add function check_unknown_options
def _signatures(
    instance: Model,
    cs: ListLike,
    es: ListLike,
    domain: str,
    peak_quant: float = PEAK_QUANT,
    max_duration: float = MAX_DURATION,
    by: str = "obs",
    **unknown_options,
):
    prcp_cvt = (
        instance.atmos_data.mean_prcp
        * 0.001
        * instance.mesh.area_dln[..., np.newaxis]
        / instance.setup.dt
    )  # convert precip from mm to m3/s

    date_range = pd.date_range(
        start=instance.setup.start_time,
        periods=prcp_cvt.shape[1],
        freq=f"{int(instance.setup.dt)}s",
    )

    col_cs = ["code"] + cs
    col_es = ["code", "season", "start", "end"] + es

    df_cs = pd.DataFrame(columns=col_cs)
    df_es = pd.DataFrame(columns=col_es)

    if len(cs) + len(es) > 0:
        for i, catchment in enumerate(instance.mesh.code):
            prcp = prcp_cvt[
                i, :
            ]  # already conversion of instance.atmos_data.mean_prcp[i, :]

            suffix = "_data" if domain == "obs" else ""
            q = getattr(instance, f"response{suffix}").q[i, :].copy()

            if (prcp < 0).all() or (q < 0).all():
                warnings.warn(
                    f"Catchment {catchment} has no precipitation or/and discharge data"
                )

                row_cs = pd.DataFrame(
                    [[catchment] + [np.nan] * (len(col_cs) - 1)], columns=col_cs
                )
                row_es = pd.DataFrame(
                    [[catchment] + [np.nan] * (len(col_es) - 1)], columns=col_es
                )

                df_cs = (
                    row_cs.copy()
                    if df_cs.empty
                    else pd.concat([df_cs, row_cs], ignore_index=True)
                )
                df_es = (
                    row_es.copy()
                    if df_es.empty
                    else pd.concat([df_es, row_es], ignore_index=True)
                )

            else:
                if len(cs) > 0:
                    csignatures = [
                        _signature_computation(prcp, q, signature) for signature in cs
                    ]

                    row_cs = pd.DataFrame([[catchment] + csignatures], columns=col_cs)

                    df_cs = (
                        row_cs.copy()
                        if df_cs.empty
                        else pd.concat([df_cs, row_cs], ignore_index=True)
                    )

                if len(es) > 0:
                    suffix = "_data" if by == "obs" else ""
                    q_seg = getattr(instance, f"response{suffix}").q[i, :].copy()

                    list_events = _events_grad(
                        prcp, q_seg, peak_quant, max_duration, instance.setup.dt
                    )

                    if len(list_events) == 0:
                        row_es = pd.DataFrame(
                            [[catchment] + [np.nan] * (len(col_es) - 1)],
                            columns=col_es,
                        )

                        df_es = (
                            row_es.copy()
                            if df_es.empty
                            else pd.concat([df_es, row_es], ignore_index=True)
                        )

                    else:
                        for t in list_events:
                            ts = t["start"]
                            te = t["end"]

                            event_prcp = prcp[ts : te + 1]
                            event_q = q[ts : te + 1]

                            season = _get_season(date_range[ts].date())

                            esignatures = [
                                _signature_computation(event_prcp, event_q, signature)
                                for signature in es
                            ]

                            row_es = pd.DataFrame(
                                [
                                    [
                                        catchment,
                                        season,
                                        date_range[ts],
                                        date_range[te],
                                    ]
                                    + esignatures
                                ],
                                columns=col_es,
                            )

                            df_es = (
                                row_es.copy()
                                if df_es.empty
                                else pd.concat([df_es, row_es], ignore_index=True)
                            )

    df_cs.replace(-99, np.nan, inplace=True)
    df_es.replace(-99, np.nan, inplace=True)

    return dict(
        zip(
            ["cont", "event"],
            [
                df_cs,
                df_es,
            ],
        )
    )


def _signature_computation(p: np.ndarray, q: np.ndarray, signature: str) -> float:
    if signature in ["Crc", "Erc"]:
        return wrap_rc(p, q)

    elif signature in ["Crchf", "Erchf"]:
        return wrap_rchf(p, q)

    elif signature in ["Crclf", "Erclf"]:
        return wrap_rclf(p, q)

    elif signature in ["Crch2r", "Erch2r"]:
        return wrap_rch2r(p, q)

    elif signature == "Cfp2":
        return wrap_cfp(q, 0.02)

    elif signature == "Cfp10":
        return wrap_cfp(q, 0.1)

    elif signature == "Cfp50":
        return wrap_cfp(q, 0.5)

    elif signature == "Cfp90":
        return wrap_cfp(q, 0.9)

    elif signature == "Eff":
        return wrap_eff(q)

    elif signature == "Ebf":
        return wrap_ebf(q)

    elif signature == "Epf":
        return wrap_epf(q)

    elif signature == "Elt":
        return wrap_elt(p, q)
