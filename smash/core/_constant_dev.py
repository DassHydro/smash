from __future__ import annotations

import numpy as np


### STRUCTURE ###
#################

STRUCTURE_PARAMETERS = {
    "gr_a": ["cp", "cft", "exc", "lr"],
    "gr_b": ["cp", "cft", "exc", "lr"],
    "gr_c": ["cp", "cft", "cst", "exc", "lr"],
    "gr_d": ["cp", "cft", "lr"],
}

STRUCTURE_STATES = {
    "gr_a": ["hp", "hft", "hlr"],
    "gr_b": ["hi", "hp", "hft", "hlr"],
    "gr_c": ["hi", "hp", "hft", "hst", "hlr"],
    "gr_d": ["hp", "hft", "hlr"],
}

STRUCTURE_COMPUTE_CI = {
    "gr_a": False,
    "gr_b": True,
    "gr_c": True,
    "gr_d": False,
}

STRUCTURE_NAME = list(STRUCTURE_PARAMETERS.keys())


### READ INPUT DATA ###
#######################

INPUT_DATA_FORMAT = ["tif", "nc"]

RATIO_PET_HOURLY = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.035,
        0.062,
        0.079,
        0.097,
        0.11,
        0.117,
        0.117,
        0.11,
        0.097,
        0.079,
        0.062,
        0.035,
        0,
        0,
        0,
        0,
        0,
    ],
    dtype=np.float32,
)
