from __future__ import annotations

import numpy as np


### MODEL STRUCTURE ###
#######################

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
# % if ci is used (depending on model structure), it will be recomputed automatically by a Fortran routine;
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


### PRECIPITATION INDICES ###
#############################

PRECIPITATION_INDICES = ["std", "d1", "d2", "vg", "hg"]


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

PY_OPTIMIZER_CLASS = ["Adam", "SGD", "Adagrad", "RMSprop"]

PY_OPTIMIZER = [opt.lower() for opt in PY_OPTIMIZER_CLASS]

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

EVENT_SEG_KEYS = ["peak_quant", "max_duration", "by"]


### GENERATE SAMPLES ###
########################

SAMPLES_GENERATORS = ["uniform", "normal", "gaussian"]

PROBLEM_KEYS = ["num_vars", "names", "bounds"]

### SIMULATION ###
##################

MAPPING = ["uniform", "distributed", "multi-linear", "multi-polynomial", "ann"]

COST_VARIANT = ["cls", "bys"]

F90_OPTIMIZER = ["sbs", "lbfgsb"]

OPTIMIZER = F90_OPTIMIZER + PY_OPTIMIZER

# % Following MAPPING order
# % The first optimizer for each mapping is used as default optimizer
MAPPING_OPTIMIZER = dict(
    zip(
        MAPPING,
        [
            F90_OPTIMIZER,
            ["lbfgsb"],
            ["lbfgsb"],
            ["lbfgsb"],
            PY_OPTIMIZER,
        ],
    )
)

F90_OPTIMIZER_CONTROL_TFM = dict(
    zip(
        F90_OPTIMIZER,
        [
            ["sbs", "normalize", "keep"],
            ["normalize", "keep"],
        ],
    )
)

JOBS_CMPT = METRICS + SIGNS

JREG_CMPT = ["prior", "smoothing", "hard-smoothing"]

WEIGHT_ALIAS = ["mean", "median", "lquartile", "uquartile"]

GAUGE_ALIAS = ["dws", "all"]

DEFAULT_TERMINATION_CRIT = dict(
    **dict(
        zip(
            F90_OPTIMIZER,
            [{"maxiter": 50}, {"maxiter": 100, "factr": 1e6, "pgtol": 1e-12}],
        )
    ),
    **dict(
        zip(
            PY_OPTIMIZER, len(PY_OPTIMIZER) * [{"epochs": 200, "early_stopping": False}]
        )
    ),
)

SIMULATION_OPTIMIZE_OPTIONS_KEYS = {
    ("uniform", "sbs"): [
        "parameters",
        "bounds",
        "control_tfm",
        "termination_crit",
    ],
    ("uniform", "lbfgsb"): [
        "parameters",
        "bounds",
        "control_tfm",
        "termination_crit",
    ],
    ("distributed", "lbfgsb"): [
        "parameters",
        "bounds",
        "control_tfm",
        "termination_crit",
    ],
    ("multi-linear", "lbfgsb"): [
        "parameters",
        "bounds",
        "control_tfm",
        "descriptor",
        "termination_crit",
    ],
    ("multi-polynomial", "lbfgsb"): [
        "parameters",
        "bounds",
        "control_tfm",
        "descriptor",
        "termination_crit",
    ],
    **dict(
        zip(
            [("ann", optimizer) for optimizer in PY_OPTIMIZER],
            len(PY_OPTIMIZER)
            * [
                [
                    "parameters",
                    "bounds",
                    "net",
                    "learning_rate",
                    "random_state",
                    "termination_crit",
                ]
            ],
        )
    ),
}

# % Following COST_VARIANT order
DEFAULT_SIMULATION_COST_OPTIONS = dict(
    zip(
        COST_VARIANT,
        [
            {
                "jobs_cmpt": "nse",
                "wjobs_cmpt": "mean",
                "wjreg": 0,
                "jreg_cmpt": None,
                "wjreg_cmpt": "mean",
                "gauge": "dws",
                "wgauge": "mean",
                "event_seg": None,
                "end_warmup": None,
            },
            {
                "jobs_cmpt": "lklhd",
                "wjobs_cmpt": "mean",
                "wjreg": 0,
                "jreg_cmpt": None,
                "wjreg_cmpt": "mean",
                "gauge": "dws",
                "event_seg": None,
                "end_warmup": None,
            },
        ],
    )
)

DEFAULT_SIMULATION_COMMON_OPTIONS = {"ncpu": 1, "verbose": True}
