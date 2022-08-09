from __future__ import annotations

from smash.solver._mwd_common import name_parameters, name_states
from smash.solver._mwd_setup import SetupDT

import yaml
import numpy as np
import os
import errno

__all__ = ["save_setup", "read_setup"]


def save_setup(setup: dict, path: str):

    """
    Save setup
    """

    with open(path, "w") as f:
        yaml.dump(setup, f, default_flow_style=False)


def read_setup(path: str) -> dict:

    """
    Read setup
    """

    if os.path.isfile(path):

        with open(path, "r") as f:

            setup = yaml.safe_load(f)

    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    return setup
