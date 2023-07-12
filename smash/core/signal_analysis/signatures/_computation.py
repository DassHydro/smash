from __future__ import annotations

from smash.core.signal_analysis.segmentation._tools import (
    _detect_peaks,
    _baseflow_separation,
)

import numpy as np


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
