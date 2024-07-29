from __future__ import annotations

import itertools

import numpy as np

### FUNCTIONS TO GENERATE CONSTANTS ###
#######################################


def get_structure() -> list[str]:
    product = itertools.product(SNOW_MODULE, HYDROLOGICAL_MODULE, ROUTING_MODULE)
    product = ["-".join(module) for module in product]

    return product


def get_rr_parameters_from_structure(structure: str) -> list[str]:
    rr_parameters = []
    [rr_parameters.extend(MODULE_RR_PARAMETERS[module]) for module in structure.split("-")]

    return rr_parameters


def get_rr_states_from_structure(structure: str) -> list[str]:
    rr_states = []
    [rr_states.extend(MODULE_RR_STATES[module]) for module in structure.split("-")]

    return rr_states


### FLOAT PRECISION FOR FLOAT COMPARISON ###
F_PRECISION = 1.0e-5
############################################


### MODULE ###
##############

SNOW_MODULE = ["zero", "ssn"]

HYDROLOGICAL_MODULE = ["gr4", "gr5", "grd", "loieau", "vic3l"]

ROUTING_MODULE = ["lag0", "lr", "kw"]

MODULE = SNOW_MODULE + HYDROLOGICAL_MODULE + ROUTING_MODULE

# % Following SNOW_MODULE order
SNOW_MODULE_RR_PARAMETERS = dict(
    zip(
        SNOW_MODULE,
        [
            [],  # % zero
            ["kmlt"],  # % ssn
        ],
    )
)

# % Following HYDROLOGICAL_MODULE order
HYDROLOGICAL_MODULE_RR_PARAMETERS = dict(
    zip(
        HYDROLOGICAL_MODULE,
        [
            ["ci", "cp", "ct", "kexc"],  # % gr4
            ["ci", "cp", "ct", "kexc", "aexc"],  # % gr5
            ["cp", "ct"],  # % grd
            ["ca", "cc", "kb"],  # % loieau
            ["b", "cusl", "cmsl", "cbsl", "ks", "pbc", "ds", "dsm", "ws"],  # % vic3l
        ],
    )
)

# % Following ROUTING_MODULE order
ROUTING_MODULE_RR_PARAMETERS = dict(
    zip(ROUTING_MODULE, [[], ["llr"], ["akw", "bkw"]])  # % lag0  # % lr  # % kw
)

# % Following MODULE order
MODULE_RR_PARAMETERS = dict(
    **SNOW_MODULE_RR_PARAMETERS,
    **HYDROLOGICAL_MODULE_RR_PARAMETERS,
    **ROUTING_MODULE_RR_PARAMETERS,
)

# % Following SNOW_MODULE order
SNOW_MODULE_RR_STATES = dict(
    zip(
        SNOW_MODULE,
        [
            [],  # % zero
            ["hs"],  # % ssn
        ],
    )
)

# % Following HYDROLOGICAL_MODULE order
HYDROLOGICAL_MODULE_RR_STATES = dict(
    zip(
        HYDROLOGICAL_MODULE,
        [
            ["hi", "hp", "ht"],  # % gr4
            ["hi", "hp", "ht"],  # % gr5
            ["hp", "ht"],  # % grd
            ["ha", "hc"],  # % loieau
            ["hcl", "husl", "hmsl", "hbsl"],  # % vic3l
        ],
    )
)

# % Following ROUTING_MODULE order
ROUTING_MODULE_RR_STATES = dict(
    zip(ROUTING_MODULE, [[], ["hlr"], []])  # % lag0  # % lr  # % kw
)

# % Following MODULE order
MODULE_RR_STATES = dict(**SNOW_MODULE_RR_STATES, **HYDROLOGICAL_MODULE_RR_STATES, **ROUTING_MODULE_RR_STATES)

# % Following ROUTING_MODULE order
ROUTING_MODULE_NQZ = dict(
    zip(ROUTING_MODULE, [1, 1, 2])  # % lag0  # % lr  # % kw
)

### STRUCTURE ###
#################

# % Product of all modules (snow, hydrological, routing)
STRUCTURE = get_structure()

# % Following STRUCTURE order
STRUCTURE_RR_PARAMETERS = dict(
    zip(
        STRUCTURE,
        [get_rr_parameters_from_structure(s) for s in STRUCTURE],
    )
)

# % Following STRUCTURE order
STRUCTURE_RR_STATES = dict(
    zip(
        STRUCTURE,
        [get_rr_states_from_structure(s) for s in STRUCTURE],
    )
)

# % Following STRUCTURE order
STRUCTURE_ADJUST_CI = dict(
    zip(
        STRUCTURE,
        ["ci" in v for v in STRUCTURE_RR_PARAMETERS.values()],
    )
)

## PARAMETERS NAME ###
######################

RR_PARAMETERS = [
    "kmlt",  # % ssn
    "ci",  # % (gr4, gr5)
    "cp",  # % (gr4, gr5, grd)
    "ct",  # % (gr4, gr5, grd)
    "kexc",  # % (gr4, gr5)
    "aexc",  # % gr5
    "ca",  # % loieau
    "cc",  # % loieau
    "kb",  # % loieau
    "b",  # % vic3l
    "cusl",  # % vic3l
    "cmsl",  # % vic3l
    "cbsl",  # % vic3l
    "ks",  # % vic3l
    "pbc",  # % vic3l
    "ds",  # % vic3l
    "dsm",  # % vic3l
    "ws",  # % vic3l
    "llr",  # % lr
    "akw",  # % kw
    "bkw",  # % kw
]

RR_STATES = [
    "hs",  # % ssn
    "hi",  # % (gr4, gr5)
    "hp",  # % (gr4, gr5, grd)
    "ht",  # % (gr4, gr5, grd)
    "ha",  # % loieau
    "hc",  # % loieau
    "hcl",  # % vic3l
    "husl",  # % vic3l
    "hmsl",  # % vic3l
    "hbsl",  # % vic3l
    "hlr",  # % lr
]

### FEASIBLE PARAMETERS ###
###########################

# % Following RR_PARAMETERS order
FEASIBLE_RR_PARAMETERS = dict(
    zip(
        RR_PARAMETERS,
        [
            (0, np.inf),  # % kmlt
            (0, np.inf),  # % ci
            (0, np.inf),  # % cp
            (0, np.inf),  # % ct
            (-np.inf, np.inf),  # % kexc
            (0, 1),  # % aexc
            (0, np.inf),  # % ca
            (0, np.inf),  # % cc
            (0, np.inf),  # % kb
            (0, np.inf),  # % b
            (0, np.inf),  # % cusl
            (0, np.inf),  # % cmsl
            (0, np.inf),  # % cbsl
            (0, np.inf),  # % ks
            (0, np.inf),  # % pbc
            (0, np.inf),  # % ds
            (0, np.inf),  # % dsm
            (0, np.inf),  # % ws
            (0, np.inf),  # % llr
            (0, np.inf),  # % akw
            (0, np.inf),  # % bkw
        ],
    )
)

# % Following RR_STATES order
FEASIBLE_RR_INITIAL_STATES = dict(
    zip(
        RR_STATES,
        [
            (0, np.inf),  # % hs
            (0, 1),  # % hi
            (0, 1),  # % hp
            (0, 1),  # % ht
            (0, 1),  # % ha
            (0, 1),  # % hc
            (0, 1),  # % hcl
            (0, 1),  # % husl
            (0, 1),  # % hmsl
            (0, 1),  # % hbsl
            (0, np.inf),  # % hlr
        ],
    )
)

### DEFAULT PARAMETERS ###
##########################

# % Following RR_PARAMETERS order
# % if ci is used (depending on model structure), it will be recomputed automatically by a Fortran routine;
# % while llr is conversed by a factor depending on the time step.
DEFAULT_RR_PARAMETERS = dict(
    zip(
        RR_PARAMETERS,
        [
            1,  # % kmlt
            1e-6,  # % ci
            200,  # % cp
            500,  # % ct
            0,  # % kexc
            0.1,  # % aexc
            200,  # % ca
            500,  # % cc
            1,  # % kb
            0.1,  # % b
            100,  # % cusl
            500,  # % cmsl
            2000,  # % cbsl
            20,  # % ks
            10,  # % pbc
            1e-2,  # % ds
            10,  # % dsm
            0.8,  # % ws
            5,  # % llr
            5,  # % akw
            0.6,  # % bkw
        ],
    )
)

# % Following RR_STATES order
DEFAULT_RR_INITIAL_STATES = dict(
    zip(
        RR_STATES,
        [
            1e-6,  # % hs
            1e-2,  # % hi
            1e-2,  # % hp
            1e-2,  # % ht
            1e-2,  # % ha
            1e-2,  # % hc
            1e-2,  # % hcl
            1e-2,  # % husl
            1e-2,  # % hmsl
            1e-6,  # % hbsl
            1e-6,  # % hlr
        ],
    )
)

### DEFAULT BOUNDS PARAMETERS ###
#################################

# % Following RR_PARAMETERS order
DEFAULT_BOUNDS_RR_PARAMETERS = dict(
    zip(
        RR_PARAMETERS,
        [
            (1e-6, 1e1),  # % kmlt
            (1e-6, 1e2),  # % ci
            (1e-6, 1e3),  # % cp
            (1e-6, 1e3),  # % ct
            (-50, 50),  # % kexc
            (1e-6, 0.999999),  # % aexc
            (1e-6, 1e3),  # % ca
            (1e-6, 1e3),  # % cc
            (1e-6, 4),  # % kb
            (1e-3, 0.8),  # % b
            (1e1, 500),  # % cusl
            (1e2, 2000),  # % cmsl
            (1e2, 2000),  # % cbsl
            (1, 1e5),  # % ks
            (1, 30),  # % pbc
            (1e-6, 0.999999),  # % ds
            (1e-3, 1e5),  # % dsm
            (1e-6, 0.999999),  # % ws
            (1e-6, 1e3),  # % llr
            (1e-3, 50),  # % akw
            (1e-3, 1),  # % bkw
        ],
    )
)

# % Following RR_STATES order
DEFAULT_BOUNDS_RR_INITIAL_STATES = dict(
    zip(
        RR_STATES,
        [
            (1e-6, 1e3),  # % hs
            (1e-6, 0.999999),  # % hi
            (1e-6, 0.999999),  # % hp
            (1e-6, 0.999999),  # % ht
            (1e-6, 0.999999),  # % ha
            (1e-6, 0.999999),  # % hc
            (1e-6, 0.999999),  # % hcl
            (1e-6, 0.999999),  # % husl
            (1e-6, 0.999999),  # % hmsl
            (1e-6, 0.999999),  # % hbsl
            (1e-6, 1e3),  # % hlr
        ],
    )
)

### OPTIMIZABLE PARAMETERS ###
##############################

# % Following RR_PARAMETERS order
OPTIMIZABLE_RR_PARAMETERS = dict(
    zip(
        RR_PARAMETERS,
        [k not in ["ci"] for k in RR_PARAMETERS],
    )
)

# % Following RR_STATES order
OPTIMIZABLE_RR_INITIAL_STATES = dict(
    zip(
        RR_STATES,
        [True] * len(RR_STATES),
    )
)

### STRUCTURAL ERROR (SERR) MODEL ###
#####################################

SERR_MU_MAPPING = ["Zero", "Constant", "Linear"]

SERR_SIGMA_MAPPING = ["Constant", "Linear", "Power", "Exponential", "Gaussian"]

SERR_MU_PARAMETERS = ["mg0", "mg1"]

SERR_SIGMA_PARAMETERS = ["sg0", "sg1", "sg2"]

# % Following SERR_MU_MAPPING order
SERR_MU_MAPPING_PARAMETERS = dict(
    zip(
        SERR_MU_MAPPING,
        [
            [],  # % zero
            ["mg0"],  # % constant
            ["mg0", "mg1"],  # % linear
        ],
    )
)

# % Following SERR_SIGMA_MAPPING order
SERR_SIGMA_MAPPING_PARAMETERS = dict(
    zip(
        SERR_SIGMA_MAPPING,
        [
            ["sg0"],  # % constant
            ["sg0", "sg1"],  # % linear
            ["sg0", "sg1", "sg2"],  # % power
            ["sg0", "sg1", "sg2"],  # % exponential
            ["sg0", "sg1", "sg2"],  # % gaussian
        ],
    )
)

### FEASIBLE SERR PARAMETERS ###
################################

# % Following SERR_MU_PARAMETERS order
FEASIBLE_SERR_MU_PARAMETERS = dict(
    zip(
        SERR_MU_PARAMETERS,
        [
            (-np.inf, np.inf),  # % mg0
            (-np.inf, np.inf),  # % mg1
        ],
    )
)

# % Following SERR_SIGMA_PARAMETERS order
FEASIBLE_SERR_SIGMA_PARAMETERS = dict(
    zip(
        SERR_SIGMA_PARAMETERS,
        [
            (0, np.inf),  # % sg0
            (0, np.inf),  # % sg1
            (0, np.inf),  # % sg2
        ],
    )
)


### DEFAULT SERR PARAMETERS ###
###############################

# % Following SERR_MU_PARAMETERS order
DEFAULT_SERR_MU_PARAMETERS = dict(
    zip(
        SERR_MU_PARAMETERS,
        [
            0,  # % mg0
            0,  # % mg1
        ],
    )
)

# % Following SERR_SIGMA_PARAMETERS order
DEFAULT_SERR_SIGMA_PARAMETERS = dict(
    zip(
        SERR_SIGMA_PARAMETERS,
        [
            1,  # % sg0
            0.2,  # % sg1
            2,  # % sg2
        ],
    )
)

### DEFAULT BOUNDS SERR PARAMETERS ###
######################################

# % Following SERR_MU_PARAMETERS order
DEFAULT_BOUNDS_SERR_MU_PARAMETERS = dict(
    zip(
        SERR_MU_PARAMETERS,
        [
            (-1e6, 1e6),  # % mg0
            (-1e6, 1e6),  # % mg1
        ],
    )
)

# % Following SERR_SIGMA_PARAMETERS order
DEFAULT_BOUNDS_SERR_SIGMA_PARAMETERS = dict(
    zip(
        SERR_SIGMA_PARAMETERS,
        [
            (1e-6, 1e3),  # % sg0
            (1e-6, 1e1),  # % sg1
            (1e-6, 1e3),  # % sg2
        ],
    )
)

### OPTIMIZABLE SERR PARAMETERS ###
###################################

# % Following SERR_MU_PARAMETERS order
OPTIMIZABLE_SERR_MU_PARAMETERS = dict(
    zip(
        SERR_MU_PARAMETERS,
        [True] * len(SERR_MU_PARAMETERS),
    )
)

# % Following SERR_SIGMA_PARAMETERS order
OPTIMIZABLE_SERR_SIGMA_PARAMETERS = dict(
    zip(
        SERR_SIGMA_PARAMETERS,
        [True] * len(SERR_SIGMA_PARAMETERS),
    )
)

### SETUP ###
#############

DEFAULT_MODEL_SETUP = {
    "snow_module": "zero",
    "hydrological_module": "gr4",
    "routing_module": "lr",
    "serr_mu_mapping": "Zero",
    "serr_sigma_mapping": "Linear",
    "dt": 3_600,
    "start_time": None,
    "end_time": None,
    "adjust_interception": True,
    "compute_mean_atmos": True,
    "read_qobs": False,
    "qobs_directory": None,
    "read_prcp": False,
    "prcp_format": "tif",
    "prcp_conversion_factor": 1,
    "prcp_directory": None,
    "prcp_access": "",
    "read_pet": False,
    "pet_format": "tif",
    "pet_conversion_factor": 1,
    "pet_directory": None,
    "pet_access": "",
    "daily_interannual_pet": False,
    "read_snow": False,
    "snow_format": "tif",
    "snow_conversion_factor": 1,
    "snow_directory": None,
    "snow_access": "",
    "read_temp": False,
    "temp_format": "tif",
    "temp_directory": None,
    "temp_access": "",
    "prcp_partitioning": False,
    "sparse_storage": False,
    "read_descriptor": False,
    "descriptor_format": "tif",
    "descriptor_directory": None,
    "descriptor_name": None,
}


### MESH ###
############

D8_VALUE = np.arange(1, 9)

### INPUT DATA ###
##################

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

REGIONAL_MAPPING = ["multi-linear", "multi-polynomial", "ann"]

MAPPING = ["uniform", "distributed"] + REGIONAL_MAPPING

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

WJREG_ALIAS = ["fast", "lcurve"]

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
    **dict(zip(PY_OPTIMIZER, len(PY_OPTIMIZER) * [{"epochs": 200, "early_stopping": 0}])),
)

CONTROL_PRIOR_DISTRIBUTION = [
    "FlatPrior",
    "Uniform",
    "Gaussian",
    "Exponential",
    "LogNormal",
    "Triangle",
]

CONTROL_PRIOR_DISTRIBUTION_PARAMETERS = dict(
    zip(
        CONTROL_PRIOR_DISTRIBUTION,
        [0, 2, 2, 2, 2, 3],
    )
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
    "forward_run": {
        "jobs_cmpt": "nse",
        "wjobs_cmpt": "mean",
        "jobs_cmpt_tfm": "keep",
        "end_warmup": None,
        "gauge": "dws",
        "wgauge": "mean",
        "event_seg": dict(zip(EVENT_SEG_KEYS[:2], [PEAK_QUANT, MAX_DURATION])),
    },
    "optimize": {
        "jobs_cmpt": "nse",
        "wjobs_cmpt": "mean",
        "jobs_cmpt_tfm": "keep",
        "wjreg": 0,
        "jreg_cmpt": "prior",
        "wjreg_cmpt": "mean",
        "end_warmup": None,
        "gauge": "dws",
        "wgauge": "mean",
        "event_seg": dict(zip(EVENT_SEG_KEYS[:2], [PEAK_QUANT, MAX_DURATION])),
    },
    "bayesian_optimize": {
        "end_warmup": None,
        "gauge": "dws",
        "control_prior": None,
    },
}

DEFAULT_SIMULATION_COMMON_OPTIONS = {"ncpu": 1, "verbose": True}

DEFAULT_SIMULATION_RETURN_OPTIONS = {
    "forward_run": {
        "time_step": "all",
        "rr_states": False,
        "q_domain": False,
        "cost": False,
        "jobs": False,
    },
    "optimize": {
        "time_step": "all",
        "rr_states": False,
        "q_domain": False,
        "iter_cost": False,
        "iter_projg": False,
        "control_vector": False,
        "net": False,
        "cost": False,
        "jobs": False,
        "jreg": False,
        "lcurve_wjreg": False,
    },
    "bayesian_optimize": {
        "time_step": "all",
        "rr_states": False,
        "q_domain": False,
        "iter_cost": False,
        "iter_projg": False,
        "control_vector": False,
        "cost": False,
        "log_lkh": False,
        "log_prior": False,
        "log_h": False,
        "serr_mu": False,
        "serr_sigma": False,
    },
    "multiset_estimate": {
        "time_step": "all",
        "rr_states": False,
        "q_domain": False,
        "cost": False,
        "jobs": False,
        "lcurve_multiset": False,
    },
}

SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS = ["rr_states", "q_domain"]

### IO ###
##########

MODEL_DDT_IO_ATTR_KEYS = {
    "setup": [
        "snow_module",
        "hydrological_module",
        "routing_module",
        "serr_mu_mapping",
        "serr_sigma_mapping",
        "start_time",
        "end_time",
        "dt",
        "descriptor_name",
    ],
    "mesh": [
        "xres",
        "yres",
        "xmin",
        "ymax",
        "dx",
        "dy",
        "active_cell",
        "gauge_pos",
        "code",
        "area",
    ],
    "response_data": ["q"],
    "physio_data": ["descriptor"],
    "atmos_data": ["mean_prcp", "mean_pet", "mean_snow", "mean_temp"],
    "rr_parameters": ["keys", "values"],
    "rr_initial_states": ["keys", "values"],
    "serr_mu_parameters": ["keys", "values"],
    "serr_sigma_parameters": ["keys", "values"],
    "response": ["q"],
    "rr_final_states": ["keys", "values"],
}
