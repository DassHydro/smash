from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.wrapping.m_setup import SetupDT

import warnings
from pandas import Timestamp


def _derived_type_parser(derived_type, data: dict):

    """
    Derived type parser
    """

    for key, value in data.items():

        if hasattr(derived_type, key):
            setattr(derived_type, key, value)

        else:
            warnings.warn(
                f"'{key}' key does not belong to the derived type {type(derived_type)}",
                UserWarning,
            )

def _standardize_setup(setup: SetupDT):
    
    """
    Check every SetupDT error/warning exception
    """

    if setup.dt < 0:
        raise ValueError("argument dt of SetupDT is lower than 0")

    if not setup.dt in [900, 3_600, 86_400]:
        warnings.warn(
            "argument dt of SetupDT is not set to a classical value (900, 3600, 86400 seconds)",
            UserWarning,
        )

    if setup.dx < 0:
        raise ValueError("argument dx of SetupDT is lower than 0")

    if setup.start_time.decode() == "":
        raise ValueError("argument start_time of SetupDT is not defined")

    if setup.end_time.decode() == "":
        raise ValueError("argument end_time of SetupDT is not defined")

    try:
        st = Timestamp(setup.start_time.decode())
    except:
        raise ValueError("argument start_time of SetupDT is not a valid date")

    try:
        et = Timestamp(setup.end_time.decode())
    except:
        raise ValueError("argument end_time of SetupDT is not a valid date")

    if (et - st).total_seconds() < 0:
        raise ValueError(
            "argument end_time of SetupDT corresponds to an earlier date than start_time"
        )

    if setup.optim_start_time.decode() == "":
        setup.optim_start_time = setup.start_time
        warnings.warn(
            "argument optim_start_time of SetupDT is not defined. Value set to start_time",
            UserWarning,
        )

    try:
        ost = Timestamp(setup.optim_start_time.decode())
    except:
        raise ValueError("argument optim_start_time of SetupDT is not a valid date")

    if (ost - st).total_seconds() < 0:
        raise ValueError(
            "argument optim_start_time of SetupDT corresponds to an earlier date than start_time"
        )

    if (et - ost).total_seconds() < 0:
        raise ValueError(
            "argument optim_start_time of SetupDT corresponds to a later date than end_time"
        )

def _build_setup(setup: SetupDT):
    
    """
    Build setup
    """
    
    _standardize_setup(setup)

    st = Timestamp(setup.start_time.decode())
    ost = Timestamp(setup.optim_start_time.decode())
    et = Timestamp(setup.end_time.decode())

    setup.nb_time_step = (et - st).total_seconds() / setup.dt

    setup.optim_start_step = (ost - st).total_seconds() / setup.dt + 1
