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


### DEFAULT PARAMETERS/STATES ###
#################################

OPR_PARAMETERS = {
    "ci": 1e-6,
    "cp": 200,
    "cft": 500,
    "cst": 500,
    "kexc": 0,
    "llr": 5,
    "akw": 5,
    "bkw": 0.6,
}  # if ci is used (depending on model structure), it will be recomputed automatically by a fortran routine; while llr is conversed by a factor depending on the timestep.

OPR_STATES = {"hi": 1e-2, "hp": 1e-2, "hft": 1e-2, "hst": 1e-2, "hlr": 1e-6}

LOW_FEASIBLE_OPR_PARAMETERS = [0, 0, 0, 0, -np.inf, 0, 0, 0]

LOW_OPTIM_OPR_PARAMETERS = [
    1e-6,
    1e-6,
    1e-6,
    1e-6,
    -50,
    1e-6,
    1e-3,
    1e-3,
]

UPP_OPTIM_OPR_PARAMETERS = [100, 1000, 1000, 10_000, 50, 1000, 50, 1]

LOW_OPR_INITIAL_STATES = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6]

UPP_OPR_INITIAL_STATES = [
    0.999999,
    0.999999,
    0.999999,
    0.999999,
    1000,
]


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


### MESH ###
############

D8_VALUE = np.arange(1, 9)


### SIGNATURES ###
##################

CSIGN = ["Crc", "Crchf", "Crclf", "Crch2r", "Cfp2", "Cfp10", "Cfp50", "Cfp90"]

ESIGN = ["Eff", "Ebf", "Erc", "Erchf", "Erclf", "Erch2r", "Elt", "Epf"]

SIGNS = CSIGN + ESIGN


### EFFICIENCY METRICS ###
##########################

EFFICIENCY_METRICS = ["nse", "kge", "se", "rmse", "logarithmic"]


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


### EVENT SEGMENTATION ###
##########################

PEAK_QUANT = 0.995

MAX_DURATION = 240


### GENERATE SAMPLES ###
########################

SAMPLE_GENERATORS = ["uniform", "normal", "gaussian"]

PROBLEM_KEYS = ["num_vars", "names", "bounds"]
