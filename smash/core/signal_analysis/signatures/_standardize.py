from __future__ import annotations

from smash._constant import SIGNS, CSIGN, ESIGN, DOMAIN

from smash.core.signal_analysis.segmentation._standardize import (
    _standardize_hydrograph_segmentation_peak_quant,
    _standardize_hydrograph_segmentation_max_duration,
    _standardize_hydrograph_segmentation_by,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash._typing import AnyTuple, ListLike

import warnings


def _standardize_signatures_sign(
    sign: str | ListLike | None,
) -> tuple[ListLike]:
    if sign is None:
        sign = SIGNS

    elif isinstance(sign, str):
        sign_standardized = sign.lower().capitalize()

        if sign_standardized not in SIGNS:
            raise ValueError(f"Unknown signature {sign}. Choices: {SIGNS}")

        else:
            sign = [sign_standardized]

    elif isinstance(sign, list):
        sign_standardized = []

        for s in sign:
            s_standardized = s.lower().capitalize()

            if s_standardized not in SIGNS:
                warnings.warn(f"Unknown signature {s}. Choices: {SIGNS}")

            else:
                sign_standardized.append(s_standardized)

        sign = sign_standardized

    else:
        raise TypeError(f"sign argument must be a str, a list of str or None")

    cs = [s for s in sign if s in CSIGN]

    es = [s for s in sign if s in ESIGN]

    return (cs, es)


def _standardize_signatures_domain(domain: str) -> str:
    if isinstance(domain, str):
        domain_standardized = domain.lower()

        if domain_standardized in DOMAIN:
            domain_standardized = domain_standardized[:3]

        else:
            raise ValueError(f"Unknown domain argument {domain}. Choices: {DOMAIN}")
    else:
        raise TypeError(f"domain argument must be str")

    return domain_standardized


def _standardize_signatures_event_seg(event_seg: dict | None) -> dict:
    if event_seg is None:
        event_seg = {}

    else:
        if isinstance(event_seg, dict):
            if "peak_quant" in event_seg:
                event_seg[
                    "peak_quant"
                ] = _standardize_hydrograph_segmentation_peak_quant(
                    event_seg["peak_quant"]
                )

            if "max_duration" in event_seg:
                event_seg[
                    "max_duration"
                ] = _standardize_hydrograph_segmentation_max_duration(
                    event_seg["max_duration"]
                )

            if "by" in event_seg:
                event_seg["by"] = _standardize_hydrograph_segmentation_by(
                    event_seg["by"]
                )

        else:
            raise TypeError(f"event_seg argument must be None or a dictionary")

    return event_seg


def _standardize_signatures_args(
    sign: str | ListLike | None,
    domain: str,
    event_seg: dict | None,
) -> AnyTuple:
    cs, es = _standardize_signatures_sign(sign)

    domain = _standardize_signatures_domain(domain)

    event_seg = _standardize_signatures_event_seg(event_seg)

    return (cs, es, domain, event_seg)
