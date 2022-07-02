from __future__ import annotations

from smash.solver._mwd_common import name_parameters, name_states
from smash.solver._mwd_setup import SetupDT

import yaml
import numpy as np
import os
import errno


SMASH_CONFIGURATION_DICT = [
    "default_parameters",
    "default_states",
    "lb_parameters",
    "ub_parameters",
    "optim_parameters",
]


def _standardize_configuration_dict(setup: SetupDT, dname: str, ditem: dict) -> list:

    if "parameters" in dname:

        default_dname = dict(zip(name_parameters, getattr(setup, dname)))

    elif "states" in dname:

        default_dname = dict(zip(name_states, getattr(setup, dname)))

    for key, value in ditem.items():

        if key in default_dname.keys():

            default_dname.update({key: value})

        else:

            raise ValueError(f"Invalid key '{key}' in '{dname}'")

    return list(default_dname.values())


def _read_yaml_configuration(path: str) -> dict:

    """
    Load yaml configuration file

    """

    if os.path.isfile(path):

        with open(path, "r") as f:

            configuration = yaml.safe_load(f)

    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    setup = SetupDT()

    for dname in SMASH_CONFIGURATION_DICT:

        if dname in configuration.keys():

            configuration[dname] = _standardize_configuration_dict(
                setup, dname, configuration[dname]
            )

    del setup

    return configuration
