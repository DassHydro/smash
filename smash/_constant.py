from __future__ import annotations

import numpy as np


### MODEL ###
#############

STRUCTURE_NAME = ["gr4-lr", "gr4-kw", "grd-lr"]

OPR_PARAMETERS = ["ci", "cp", "ct", "kexc", "llr", "akw", "bkw"]

OPR_STATES = ["hi", "hp", "ht", "hlr"]

# % Following STRUCTURE_NAME order
STRUCTURE_OPR_PARAMETERS = dict(
    zip(
        STRUCTURE_NAME,
        [
            ["ci", "cp", "ct", "kexc", "llr"],
            ["ci", "cp", "ct", "kexc", "akw", "bkw"],
            ["cp", "ct", "llr"],
        ],
    )
)

# % Following STRUCTURE_NAME order
STRUCTURE_OPR_STATES = dict(
    zip(
        STRUCTURE_NAME,
        [
            ["hi", "hp", "ht", "hlr"],
            ["hi", "hp", "ht"],
            ["hp", "ht", "hlr"],
        ],
    )
)

# % Following STRUCTURE_NAME order
STRUCTURE_COMPUTE_CI = dict(zip(STRUCTURE_NAME, [True, True, False]))

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
            (0, np.inf),
        ],
    )
)

### DEFAULT PARAMETERS ###
##########################

# % Following OPR_PARAMETERS order
# % if ci is used (depending on model structure), it will be recomputed automatically by a fortran routine;
# % while llr is conversed by a factor depending on the timestep.
DEFAULT_OPR_PARAMETERS = dict(zip(OPR_PARAMETERS, [1e-6, 200, 500, 0, 5, 5, 0.6]))

# % Following OPR_STATES order
DEFAULT_OPR_INITIAL_STATES = dict(zip(OPR_STATES, [1e-2, 1e-2, 1e-2, 1e-6]))

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
            (1e-6, 1e3),
        ],
    )
)

TOL_BOUNDS = 1e-9

### OPTIMIZABLE PARAMETERS ###
##############################

# % Following OPR_PARAMETERS order
OPTIMIZABLE_OPR_PARAMETERS = dict(
    zip(
        OPR_PARAMETERS,
        [False, True, True, True, True, True, True],
    )
)

# % Following OPR_STATES order
OPTIMIZABLE_OPR_INITIAL_STATES = dict(
    zip(
        OPR_STATES,
        [True, True, True, True],
    )
)

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

### DATASET ###
###############

DATASET_NAME = ["flwdir", "cance", "lez", "france"]

### MESH ###
############

D8_VALUE = np.arange(1, 9)


### SIGNATURES ###
##################

CSIGN = ["Crc", "Crchf", "Crclf", "Crch2r", "Cfp2", "Cfp10", "Cfp50", "Cfp90"]

ESIGN = ["Eff", "Ebf", "Erc", "Erchf", "Erclf", "Erch2r", "Elt", "Epf"]

SIGNS = CSIGN + ESIGN

DOMAIN = ["obs", "observed", "observation", "sim", "simulated", "simulation"]


### EFFICIENCY/ERROR METRICS ###
################################

METRICS = ["nse", "nnse", "kge", "mae", "mape", "mse", "rmse", "lgrm"]


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

LAYER_NAME = ["Dense", "Activation", "Scale", "Dropout"]

NET_OPTIMIZER = ["SGD", "Adam", "Adagrad", "RMSprop"]

ACTIVATION_FUNCTION = [
    "Sigmoid",
    "Softmax",
    "TanH",
    "ReLU",
    "LeakyReLU",
    "ELU",
    "SELU",
    "SoftPlus",
]


### EVENT SEGMENTATION ###
##########################

PEAK_QUANT = 0.995

MAX_DURATION = 240


### GENERATE SAMPLES ###
########################

SAMPLES_GENERATORS = ["uniform", "normal", "gaussian"]

PROBLEM_KEYS = ["num_vars", "names", "bounds"]
