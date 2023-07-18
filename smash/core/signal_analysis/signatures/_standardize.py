from __future__ import annotations

from smash._constant import SIGNS, CSIGN, ESIGN, COMPUTE_BY

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash._typing import AnyTuple

import warnings


def _standardize_compute_signatures_sign(
    sign: str | list[str] | None,
) -> tuple[list[str]]:
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


def _standardize_compute_signatures_event_seg(event_seg: dict | None) -> dict:
    if event_seg is None:
        event_seg = {}

    return event_seg


def _standardize_compute_signatures_by(by: str) -> str:
    if isinstance(by, str):
        by_standardized = by.lower()

        if by_standardized in COMPUTE_BY:
            by_standardized = by_standardized[:3]

        else:
            raise ValueError(f"Unknown by argument {by}. Choices: {COMPUTE_BY}")
    else:
        raise TypeError(f"by argument must be str")

    return by_standardized


def _standardize_compute_signatures_args(
    sign: str | list[str] | None,
    event_seg: dict | None,
    by: str,
) -> AnyTuple:
    cs, es = _standardize_compute_signatures_sign(sign)

    event_seg = _standardize_compute_signatures_event_seg(event_seg)

    by = _standardize_compute_signatures_by(by)

    return (cs, es, event_seg, by)
