from __future__ import annotations

from smash.solver._mwd_options import OptionsDT
from smash.solver._mwd_returns import ReturnsDT

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver._mwd_setup import SetupDT


def _standardize_options(options: OptionsDT | None, setup: SetupDT):
    if options is None:
        options = OptionsDT(setup)

    elif isinstance(options, OptionsDT):
        ...

    else:
        raise TypeError(f"options argument must be None or of type OptionsDT")

    return options


def _standardize_returns(returns: ReturnsDT):
    if returns is None:
        returns = ReturnsDT()

    elif isinstance(returns, ReturnsDT):
        ...

    else:
        raise TypeError(f"returns argument must be None or of type ReturnsDT")

    return returns
