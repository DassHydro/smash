from __future__ import annotations

import numpy as np


### STRUCTURE ###
#################

STRUCTURE_OPR_PARAMETERS = {
    "gr-a-lr": ["cp", "cft", "kexc", "llr"],
    "gr-b-lr": ["ci", "cp", "cft", "kexc", "llr"],
    "gr-c-lr": ["ci", "cp", "cft", "cst", "kexc", "llr"],
    "gr-d-lr": ["cp", "cft", "lr"],
    "gr-a-kw": ["cp", "cft", "kexc", "akw", "bkw"],
}

STRUCTURE_OPR_STATES = {
    "gr-a-lr": ["hp", "hft", "hlr"],
    "gr-b-lr": ["hi", "hp", "hft", "hlr"],
    "gr-c-lr": ["hi", "hp", "hft", "hst", "hlr"],
    "gr-d-lr": ["hp", "hft", "hlr"],
    "gr-a-kw": ["hp", "hft"],
}

STRUCTURE_COMPUTE_CI = {
    "gr-a-lr": False,
    "gr-b-lr": True,
    "gr-c-lr": True,
    "gr-d-lr": False,
    "gr-a-kw": False,
}

STRUCTURE_NAME = list(STRUCTURE_OPR_PARAMETERS.keys())


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


### NET ###
###########

WB_INITIALIZER = [
    "uniform",
    "glorot_uniform",
    "he_uniform",
    "normal",
    "glorot_normal",
    "he_normal",
    "zeros",
]

NET_OPTIMIZER = ["sgd", "adam", "adagrad", "rmsprop"]

LAYER_NAME = ["dense", "activation", "scale", "dropout"]