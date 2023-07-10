from __future__ import annotations

import numpy as np


### MODEL ###
#############

STRUCTURE_NAME = ["gr-a-lr", "gr-b-lr", "gr-c-lr", "gr-d-lr", "gr-a-kw"]

OPR_PARAMETERS = ["ci", "cp", "cft", "cst", "kexc", "llr", "akw", "bkw"]

OPR_STATES = ["hi", "hp", "hft", "hst", "hlr"]

# % Following STRUCTURE_NAME order
STRUCTURE_OPR_PARAMETERS = dict(
    zip(
        STRUCTURE_NAME,
        [
            ["cp", "cft", "kexc", "llr"],
            ["ci", "cp", "cft", "kexc", "llr"],
            ["ci", "cp", "cft", "cst", "kexc", "llr"],
            ["cp", "cft", "llr"],
            ["cp", "cft", "kexc", "akw", "bkw"],
        ],
    )
)

# % Following STRUCTURE_NAME order
STRUCTURE_OPR_STATES = dict(
    zip(
        STRUCTURE_NAME,
        [
            ["hp", "hft", "hlr"],
            ["hi", "hp", "hft", "hlr"],
            ["hi", "hp", "hft", "hst", "hlr"],
            ["hp", "hft", "hlr"],
            ["hp", "hft"],
        ],
    )
)

# % Following STRUCTURE_NAME order
STRUCTURE_COMPUTE_CI = dict(zip(STRUCTURE_NAME, [False, True, True, False, False]))

### FEASIBLE PARAMETERS ###
###########################

# % Following OPR_PARAMETERS order
FEASIBLE_OPR_PARAMETERS = dict(
    zip(
        OPR_PARAMETERS,
        [
            (0, np.inf),
            (0, np.inf),
            (0, np.inf),
            (0, np.inf),
            (-np.inf, np.inf),
            (0, np.inf),
            (0, np.inf),
            (0, np.inf),
        ],
    )
)

# % Following OPR_STATES order
FEASIBLE_OPR_INITIAL_STATES = dict(
    zip(
        OPR_STATES,
        [
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, np.inf),
        ],
    )
)

### DEFAULT PARAMETERS ###
##########################

# % Following OPR_PARAMETERS order
# % if ci is used (depending on model structure), it will be recomputed automatically by a fortran routine;
# % while llr is conversed by a factor depending on the timestep.
DEFAULT_OPR_PARAMETERS = dict(zip(OPR_PARAMETERS, [1e-6, 200, 500, 500, 0, 5, 5, 0.6]))

# % Following OPR_STATES order
DEFAULT_OPR_INITIAL_STATES = dict(zip(OPR_STATES, [1e-2, 1e-2, 1e-2, 1e-2, 1e-6]))

### DEFAULT BOUNDS PARAMETERS ###
#################################

# % Following OPR_PARAMETERS order
DEFAULT_BOUNDS_OPR_PARAMETERS = dict(
    zip(
        OPR_PARAMETERS,
        [
            (1e-6, 1e2),
            (1e-6, 1e3),
            (1e-6, 1e3),
            (1e-6, 1e4),
            (-50, 50),
            (1e-6, 1e3),
            (1e-3, 50),
            (1e-3, 1),
        ],
    )
)

# % Following OPR_STATES order
DEFAULT_BOUNDS_OPR_INITIAL_STATES = dict(
    zip(
        OPR_STATES,
        [
            (1e-6, 0.999999),
            (1e-6, 0.999999),
            (1e-6, 0.999999),
            (1e-6, 0.999999),
            (1e-6, 1e3),
        ],
    )
)

TOL_BOUNDS = 1e-9


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

SAMPLES_GENERATORS = ["uniform", "normal", "gaussian"]

PROBLEM_KEYS = ["num_vars", "names", "bounds"]
