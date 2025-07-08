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


def get_rr_internal_fluxes_from_structure(structure: str) -> list[str]:
    rr_internal_fluxes = []
    [rr_internal_fluxes.extend(MODULE_RR_INTERNAL_FLUXES[module]) for module in structure.split("-")]
    return rr_internal_fluxes


def get_neurons_from_hydrological_module(hydrological_module: str, hidden_neuron: np.ndarray) -> np.ndarray:
    n_in, n_out = HYDROLOGICAL_MODULE_INOUT_NEURONS[hydrological_module]
    neurons = [n_in] + [val for val in hidden_neuron if val > 0] + [n_out]
    padded_neurons = np.zeros(len(hidden_neuron) + 2, dtype=np.int32)
    padded_neurons[: len(neurons)] = neurons

    return padded_neurons


### FLOAT PRECISION FOR FLOAT COMPARISON ###
F_PRECISION = 1.0e-5

### MODULE ###
##############

SNOW_MODULE = ["zero", "ssn"]

HYDROLOGICAL_MODULE = [
    "gr4",
    "gr4_mlp",
    "gr4_ri",
    "gr4_ode",
    "gr4_ode_mlp",
    "gr5",
    "gr5_mlp",
    "gr5_ri",
    "gr6",
    "gr6_mlp",
    "grc",
    "grc_mlp",
    "grd",
    "grd_mlp",
    "loieau",
    "loieau_mlp",
    "vic3l",
]

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
        (
            [["ci", "cp", "ct", "kexc"]] * 2  # % gr4, gr4_mlp,
            + [["ci", "cp", "ct", "alpha1", "alpha2", "kexc"]]  # % gr4_ri
            + [["ci", "cp", "ct", "kexc"]] * 2  # % gr4_ode, gr4_ode_mlp
            + [["ci", "cp", "ct", "kexc", "aexc"]] * 2  # % gr5, gr5_mlp
            + [["ci", "cp", "ct", "alpha1", "alpha2", "kexc", "aexc"]]  # % gr5_ri
            + [["ci", "cp", "ct", "be", "kexc", "aexc"]] * 2  # % gr6, gr6_mlp
            + [["ci", "cp", "ct", "cl", "kexc"]] * 2  # % grc, grc_mlp
            + [["cp", "ct"]] * 2  # % grd, grd_mlp
            + [["ca", "cc", "kb"]] * 2  # % loieau, loieau_mlp
            + [["b", "cusl", "cmsl", "cbsl", "ks", "pbc", "ds", "dsm", "ws"]]  # % vic3l
        ),
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
        (
            [["hi", "hp", "ht"]] * 8  # % gr4, gr4_mlp, gr4_ri, gr4_ode, gr4_ode_mlp, gr5, gr5_mlp, gr5_ri
            + [["hi", "hp", "ht", "he"]] * 2  # % gr6, gr6_mlp
            + [["hi", "hp", "ht", "hl"]] * 2  # % grc, grc_mlp
            + [["hp", "ht"]] * 2  # % grd, grd_mlp
            + [["ha", "hc"]] * 2  # % loieau, loieau_mlp
            + [["hcl", "husl", "hmsl", "hbsl"]]  # % vic3l
        ),
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

# % Following SNOW_MODULE order
SNOW_MODULE_RR_INTERNAL_FLUXES = dict(
    zip(
        SNOW_MODULE,
        [
            [],  # % zero
            ["mlt"],  # % ssn
        ],
    )
)

# % Following HYDROLOGICAL_MODULE order
HYDROLOGICAL_MODULE_RR_INTERNAL_FLUXES = dict(
    zip(
        HYDROLOGICAL_MODULE,
        (
            [["pn", "en", "pr", "perc", "ps", "es", "lexc", "prr", "prd", "qr", "qd", "qt"]]
            * 3  # % gr4, gr4_mlp, gr4_ri
            + [["pn", "en", "lexc", "qt"]] * 2  # % gr4_ode, gr4_ode_mlp
            + [["pn", "en", "pr", "perc", "ps", "es", "lexc", "prr", "prd", "qr", "qd", "qt"]]
            * 3  # % gr5, gr5_mlp, gr5_ri
            + [["pn", "en", "pr", "perc", "ps", "es", "lexc", "prr", "prd", "pre", "qr", "qd", "qe", "qt"]]
            * 2  # % gr6, gr6_mlp
            + [["pn", "en", "pr", "perc", "ps", "es", "lexc", "prr", "prd", "prl", "qr", "qd", "ql", "qt"]]
            * 2  # % grc, grc_mlp
            + [["ei", "pn", "en", "pr", "perc", "ps", "es", "prr", "qr", "qt"]] * 2  # % grd, grd_mlp
            + [["ei", "pn", "en", "pr", "perc", "ps", "es", "prr", "prd", "qr", "qd", "qt"]]
            * 2  # % loieau, loieau_mlp
            + [["pn", "en", "qr", "qb", "qt"]]  # % vic3l
        ),
    )
)

# % Following ROUTING_MODULE order
ROUTING_MODULE_RR_INTERNAL_FLUXES = dict(
    zip(ROUTING_MODULE, [["qup"], ["qup"], ["qim1j"]])  # % lag0  # % lr  # % kw
)

MODULE_RR_INTERNAL_FLUXES = dict(
    **SNOW_MODULE_RR_INTERNAL_FLUXES,
    **HYDROLOGICAL_MODULE_RR_INTERNAL_FLUXES,
    **ROUTING_MODULE_RR_INTERNAL_FLUXES,
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

# % Following STRUCTURE order
STRUCTURE_RR_INTERNAL_FLUXES = dict(
    zip(
        STRUCTURE,
        [get_rr_internal_fluxes_from_structure(s) for s in STRUCTURE],
    )
)

## PARAMETERIZATION NN STRUCTURE ##
###################################

# % Following HYDROLOGICAL_MODULE order
HYDROLOGICAL_MODULE_INOUT_NEURONS = dict(
    zip(
        HYDROLOGICAL_MODULE,
        (
            [
                (0, 0),  # % gr4
                (4, 4),  # % gr4_mlp
                (0, 0),  # % gr4_ri
                (0, 0),  # % gr4_ode
                (4, 4),  # % gr4_ode_mlp
                (0, 0),  # % gr5
                (4, 4),  # % gr5_mlp
                (0, 0),  # % gr5_ri
                (0, 0),  # % gr6
                (5, 5),  # % gr6_mlp
                (0, 0),  # % grc
                (5, 5),  # % grc_mlp
                (0, 0),  # % grd
                (4, 2),  # % grd_mlp
                (0, 0),  # % loieau
                (4, 3),  # % loieau_mlp
                (0, 0),  # % vic3l
            ]
        ),
    )
)

## PARAMETERS NAME ###
######################

RR_PARAMETERS = [
    "kmlt",  # % ssn
    "ci",  # % gr4, gr4_mlp, gr4_ri, gr4_ode, gr4_ode_mlp, gr5, gr5_mlp, gr5_ri, gr6, gr6_mlp, grc, grc_mlp
    "cp",  # % gr4, gr4_mlp, gr4_ri, gr4_ode, gr4_ode_mlp, gr5, gr5_mlp, gr5_ri, gr6, gr6_mlp, grc, grc_mlp,
    # % grd, grd_mlp
    "ct",  # % gr4, gr4_mlp, gr4_ri, gr4_ode, gr4_ode_mlp, gr5, gr5_mlp, gr5_ri, gr6, gr6_mlp, grc, grc_mlp,
    # %grd, grd_mlp
    "alpha1",  # % gr4_ri, gr5_ri
    "alpha2",  # % gr4_ri, gr5_ri
    "cl",  # % grc, grc_mlp
    "be",  # % gr6, gr6_mlp
    "kexc",  # % gr4, gr4_mlp, gr4_ri, gr4_ode, gr4_ode_mlp, gr5, gr5_mlp, gr6, gr6_mlp, grc, grc_mlp
    "aexc",  # % gr5, gr5_mlp, gr5_ri, gr6, gr6_mlp
    "ca",  # % loieau, loieau_mlp
    "cc",  # % loieau, loieau_mlp
    "kb",  # % loieau, loieau_mlp
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
    "hi",  # % gr4, gr4_mlp, gr4_ri, gr4_ode, gr4_ode_mlp, gr5, gr5_mlp, gr5_ri, gr6, gr6_mlp, grc, grc_mlp
    "hp",  # % gr4, gr4_mlp, gr4_ri, gr4_ode, gr4_ode_mlp, gr5, gr5_mlp, gr5_ri, gr6, gr6_mlp, grc, grc_mlp,
    # % grd, grd_mlp
    "ht",  # % gr4, gr4_mlp, gr4_ri, gr4_ode, gr4_ode_mlp, gr5, gr5_mlp, gr5_ri, gr6, gr6_mlp, grc, grc_mlp,
    # % grd, grd_mlp
    "hl",  # % grc, grc_mlp
    "he",  # % gr6, gr6_mlp
    "ha",  # % loieau, loieau_mlp
    "hc",  # % loieau, loieau_mlp
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
            (0, np.inf),  # % alpha1
            (0, np.inf),  # % alpha2
            (0, np.inf),  # % cl
            (0, np.inf),  # % be
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
            (0, 1),  # % hl
            (-np.inf, np.inf),  # % he
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
            3e-4,  # % alpha1
            1e-3,  # % alpha2
            500,  # % cl
            10,  # % be
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
            1e-2,  # % hl
            -100,  # % he
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
            (1e-6, 5e-4),  # % alpha1
            (1e-5, 1e-3),  # % alpha2
            (1e-6, 1e3),  # % cl
            (1e-3, 20),  # % be
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
            (1e-6, 0.999999),  # % hl
            (-1e3, 0),  # % he
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

### OPTIMIZABLE NN PARAMETERS ###
#################################

NN_PARAMETERS_KEYS = ["weight_1", "bias_1", "weight_2", "bias_2", "weight_3", "bias_3"]

OPTIMIZABLE_NN_PARAMETERS = [
    [],  # without hybrid structure
    NN_PARAMETERS_KEYS[:-2],  # minimal hybrid structure with 2 layers
    NN_PARAMETERS_KEYS,  # maximal hybrid structure with 3 layers
]

### SETUP ###
#############

DEFAULT_MODEL_SETUP = {
    "snow_module": "zero",
    "hydrological_module": "gr4",
    "routing_module": "lr",
    "hidden_neuron": 16,
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
    "read_imperviousness": False,
    "imperviousness_format": "tif",
    "imperviousness_file": None,
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

OPTIMIZER_CLASS = ["Adam", "SGD", "Adagrad", "RMSprop"]

ACTIVATION_FUNCTION_CLASS = [
    "Sigmoid",
    "Softmax",
    "TanH",
    "ReLU",
    "LeakyReLU",
    "ELU",
    "SELU",
    "SoftPlus",
]

ACTIVATION_FUNCTION = [func.lower() for func in ACTIVATION_FUNCTION_CLASS]


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

REGIONAL_MAPPING = ["multi-linear", "multi-power", "ann"]

MAPPING = ["uniform", "distributed"] + REGIONAL_MAPPING

ADAPTIVE_OPTIMIZER = [opt.lower() for opt in OPTIMIZER_CLASS]
GRADIENT_BASED_OPTIMIZER = ["lbfgsb"] + ADAPTIVE_OPTIMIZER
GRADIENT_FREE_OPTIMIZER = ["sbs", "nelder-mead", "powell"]

OPTIMIZER = GRADIENT_FREE_OPTIMIZER + GRADIENT_BASED_OPTIMIZER

# % Following MAPPING order
# % The first optimizer for each mapping is used as default optimizer
MAPPING_OPTIMIZER = dict(
    zip(
        MAPPING,
        [
            OPTIMIZER,  # for uniform mapping (all optimizers are possible, default is sbs)
            *(
                [GRADIENT_BASED_OPTIMIZER] * 3
            ),  # for distributed, multi-linear, multi-power mappings (default is lbfgsb)
            ADAPTIVE_OPTIMIZER,  # for ann mapping (default is adam)
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
            ["sbs", "nelder-mead", "powell", "lbfgsb"],
            [
                {"maxiter": 50},
                {"maxiter": 200, "xatol": 1e-4, "fatol": 1e-4},
                {"maxiter": 50},
                {"maxiter": 100, "factr": 1e6, "pgtol": 1e-12},
            ],
        )
    ),
    **dict(zip(ADAPTIVE_OPTIMIZER, len(ADAPTIVE_OPTIMIZER) * [{"maxiter": 200, "early_stopping": 0}])),
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
    ("uniform", "nelder-mead"): [
        "parameters",
        "bounds",
        "control_tfm",
        "termination_crit",
    ],
    ("uniform", "powell"): [
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
    **dict(
        zip(
            itertools.product(["uniform", "distributed"], ADAPTIVE_OPTIMIZER),
            2
            * len(ADAPTIVE_OPTIMIZER)
            * [
                [
                    "parameters",
                    "bounds",
                    "control_tfm",
                    "learning_rate",
                    "termination_crit",
                ]
            ],
        )
    ),  # product between 2 mappings (uniform, distributed) and all adaptive optimizers
    ("multi-linear", "lbfgsb"): [
        "parameters",
        "bounds",
        "control_tfm",
        "descriptor",
        "termination_crit",
    ],
    ("multi-power", "lbfgsb"): [
        "parameters",
        "bounds",
        "control_tfm",
        "descriptor",
        "termination_crit",
    ],
    **dict(
        zip(
            itertools.product(["multi-linear", "multi-power"], ADAPTIVE_OPTIMIZER),
            2
            * len(ADAPTIVE_OPTIMIZER)
            * [
                [
                    "parameters",
                    "bounds",
                    "control_tfm",
                    "descriptor",
                    "learning_rate",
                    "termination_crit",
                ]
            ],
        )
    ),  # product between 2 mappings (multi-linear, multi-power) and all adaptive optimizers
    **dict(
        zip(
            [("ann", optimizer) for optimizer in ADAPTIVE_OPTIMIZER],
            len(ADAPTIVE_OPTIMIZER)
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
    ),  # ann mapping and all adaptive optimizers
}

OPTIMIZER_CONTROL_TFM = {
    (mapping, optimizer): ["sbs", "normalize", "keep"]  # in case of sbs optimizer
    if optimizer == "sbs"
    else ["normalize", "keep"]  # for other optimizers (not used with ann mapping)
    if mapping != "ann"
    else ["keep"]  # no tfm applied for any optimizer used with ann mapping
    for mapping, optimizer in SIMULATION_OPTIMIZE_OPTIONS_KEYS.keys()
}  # first element of the list is the default tfm for each tuple key (mapping, optimizer)

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
        "internal_fluxes": False,
        "cost": False,
        "jobs": False,
    },
    "optimize": {
        "time_step": "all",
        "rr_states": False,
        "q_domain": False,
        "internal_fluxes": False,
        "control_vector": False,
        "net": False,
        "cost": False,
        "n_iter": False,
        "projg": False,
        "jobs": False,
        "jreg": False,
        "lcurve_wjreg": False,
    },
    "bayesian_optimize": {
        "time_step": "all",
        "rr_states": False,
        "q_domain": False,
        "internal_fluxes": False,
        "control_vector": False,
        "cost": False,
        "n_iter": False,
        "projg": False,
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
        "internal_fluxes": False,
        "cost": False,
        "jobs": False,
        "lcurve_multiset": False,
    },
}

SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS = ["rr_states", "q_domain", "internal_fluxes"]

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
    "nn_parameters": NN_PARAMETERS_KEYS,
    "serr_mu_parameters": ["keys", "values"],
    "serr_sigma_parameters": ["keys", "values"],
    "response": ["q"],
    "rr_final_states": ["keys", "values"],
}
