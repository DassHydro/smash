from __future__ import annotations

import numpy as np


### MODEL STRUCTURE ###
#######################

STRUCTURE_NAME = ["gr4-lr", "gr4-kw", "gr5-lr", "gr5-kw", "loieau-lr", "grd-lr"]

OPR_PARAMETERS = [
    "ci",
    "cp",
    "ct",
    "kexc",
    "aexc",
    "ca",
    "cc",
    "kb",
    "llr",
    "akw",
    "bkw",
]

OPR_STATES = ["hi", "hp", "ht", "ha", "hc", "hlr"]

# % Following STRUCTURE_NAME order
STRUCTURE_OPR_PARAMETERS = dict(
    zip(
        STRUCTURE_NAME,
        [
            ["ci", "cp", "ct", "kexc", "llr"],  # % gr4-lr
            ["ci", "cp", "ct", "kexc", "akw", "bkw"],  # % gr4-kw
            ["ci", "cp", "ct", "kexc", "aexc", "llr"],  # % gr5-lr
            ["ci", "cp", "ct", "kexc", "aexc", "akw", "bkw"],  # % gr4-kw
            ["ca", "cc", "kb", "llr"],  # % loieau-lr
            ["cp", "ct", "llr"],  # % grd-lr
        ],
    )
)

# % Following STRUCTURE_NAME order
STRUCTURE_OPR_STATES = dict(
    zip(
        STRUCTURE_NAME,
        [
            ["hi", "hp", "ht", "hlr"],  # % gr4-lr
            ["hi", "hp", "ht"],  # % gr4-kw
            ["hi", "hp", "ht", "hlr"],  # % gr5-lr
            ["hi", "hp", "ht"],  # % gr5-kw
            ["ha", "hc", "hlr"],  # % loieau-lr
            ["hp", "ht", "hlr"],  # % grd-lr
        ],
    )
)

# % Following STRUCTURE_NAME order
STRUCTURE_COMPUTE_CI = dict(
    zip(
        STRUCTURE_NAME,
        ["ci" in v for v in STRUCTURE_OPR_PARAMETERS.values()],
    )
)

### FEASIBLE PARAMETERS ###
###########################

# % Following OPR_PARAMETERS order
FEASIBLE_OPR_PARAMETERS = dict(
    zip(
        OPR_PARAMETERS,
        [
            (0, np.inf),  # % ci
            (0, np.inf),  # % cp
            (0, np.inf),  # % ct
            (-np.inf, np.inf),  # % kexc
            (0, 1),  # % aexc
            (0, np.inf),  # % ca
            (0, np.inf),  # % cc
            (0, np.inf),  # % kb
            (0, np.inf),  # % llr
            (0, np.inf),  # % akw
            (0, np.inf),  # % bkw
        ],
    )
)

# % Following OPR_STATES order
FEASIBLE_OPR_INITIAL_STATES = dict(
    zip(
        OPR_STATES,
        [
            (0, 1),  # % hi
            (0, 1),  # % hp
            (0, 1),  # % ht
            (0, 1),  # % ha
            (0, 1),  # % hc
            (0, np.inf),  # % hlr
        ],
    )
)

### DEFAULT PARAMETERS ###
##########################

# % Following OPR_PARAMETERS order
# % if ci is used (depending on model structure), it will be recomputed automatically by a Fortran routine;
# % while llr is conversed by a factor depending on the timestep.
DEFAULT_OPR_PARAMETERS = dict(
    zip(
        OPR_PARAMETERS,
        [
            1e-6,  # % ci
            200,  # % cp
            500,  # % ct
            0,  # % kexc
            0.1,  # % aexc
            200,  # % ca
            500,  # % cc
            1,  # % kb
            5,  # % llr
            5,  # % akw
            0.6,  # % bkw
        ],
    )
)

# % Following OPR_STATES order
DEFAULT_OPR_INITIAL_STATES = dict(
    zip(
        OPR_STATES,
        [
            1e-2,  # % hi
            1e-2,  # % hp
            1e-2,  # % ht
            1e-2,  # % ha
            1e-2,  # % hc
            1e-6,  # % hlr
        ],
    )
)

### DEFAULT BOUNDS PARAMETERS ###
#################################

# % Following OPR_PARAMETERS order
DEFAULT_BOUNDS_OPR_PARAMETERS = dict(
    zip(
        OPR_PARAMETERS,
        [
            (1e-6, 1e2),  # % ci
            (1e-6, 1e3),  # % cp
            (1e-6, 1e3),  # % ct
            (-50, 50),  # % kexc
            (1e-6, 0.999999),  # % aexc
            (1e-6, 1e3),  # % ca
            (1e-6, 1e3),  # % cc
            (1e-6, 4),  # % kb
            (1e-6, 1e3),  # % llr
            (1e-3, 50),  # % akw
            (1e-3, 1),  # % bkw
        ],
    )
)

# % Following OPR_STATES order
DEFAULT_BOUNDS_OPR_INITIAL_STATES = dict(
    zip(
        OPR_STATES,
        [
            (1e-6, 0.999999),  # % hi
            (1e-6, 0.999999),  # % hp
            (1e-6, 0.999999),  # % ht
            (1e-6, 0.999999),  # % ha
            (1e-6, 0.999999),  # % hc
            (1e-6, 1e3),  # % hlr
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
        [k not in ["ci"] for k in OPR_PARAMETERS],
    )
)

# % Following OPR_STATES order
OPTIMIZABLE_OPR_INITIAL_STATES = dict(
    zip(
        OPR_STATES,
        [True] * len(OPR_STATES),
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

JOBS_CMPT_TFM = ["keep", "sqrt", "inv"]

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
        zip(PY_OPTIMIZER, len(PY_OPTIMIZER) * [{"epochs": 200, "early_stopping": 0}])
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

DEFAULT_SIMULATION_COST_OPTIONS = {
    "jobs_cmpt": "nse",
    "wjobs_cmpt": "mean",
    "jobs_cmpt_tfm": None,
    "wjreg": 0,
    "jreg_cmpt": "prior",
    "wjreg_cmpt": "mean",
    "gauge": "dws",
    "wgauge": "mean",
    "event_seg": dict(zip(EVENT_SEG_KEYS[:2], [PEAK_QUANT, MAX_DURATION])),
    "end_warmup": None,
}

DEFAULT_SIMULATION_COMMON_OPTIONS = {"ncpu": 1, "verbose": True}

DEFAULT_SIMULATION_RETURN_OPTIONS = {
    "forward_run": {
        "opr_states": False,
        "q_domain": False,
        "cost": False,
        "jobs": False,
        "time_step": "all",
    },
    "optimize": {
        "opr_states": False,
        "q_domain": False,
        "iter_cost": False,
        "iter_projg": False,
        "control_vector": False,
        "net": False,
        "cost": False,
        "jobs": False,
        "jreg": False,
        "lcurve_wjreg": False,
        "time_step": "all",
    },
    "multiset_estimate": {
        "opr_states": False,
        "q_domain": False,
        "cost": False,
        "jobs": False,
        "lcurve_multiset": False,
        "time_step": "all",
    },
}

SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS = ["opr_states", "q_domain"]
