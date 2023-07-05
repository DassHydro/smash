from __future__ import annotations

from smash._constant import SIGNS


def _standardize_signatures(sign: str | list[str] | None):
    if sign is None:
        sign = SIGNS

    elif isinstance(sign, str):
        if sign not in SIGNS:
            raise ValueError(f"Unknown signature {sign}. Choices: {SIGNS}")

        else:
            sign = [sign]

    elif isinstance(sign, list):
        unk_sign = tuple(s for s in sign if s not in SIGNS)

        if unk_sign:
            raise ValueError(f"Unknown signature(s) {unk_sign}. Choices: {SIGNS}")

    else:
        raise TypeError(f"Signature(s) must be a str, a list of str or None")

    cs = [s for s in sign if s[0] == "C"]

    es = [s for s in sign if s[0] == "E"]

    return cs, es
