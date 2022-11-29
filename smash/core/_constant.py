from __future__ import annotations

import numpy as np


### GLOBAL ###
##############

STRUCTURE_PARAMETERS = {
    "gr-a": ["cp", "cft", "exc", "lr"],
    "gr-b": ["ci", "cp", "cft", "cst", "exc", "lr"],
    "vic-a": ["b", "cusl1", "cusl2", "clsl", "ks", "ds", "dsm", "ws", "lr"],
}

STRUCTURE_STATES = {
    "gr-a": ["hp", "hft", "hlr"],
    "gr-b": ["hi", "hp", "hft", "hst", "hlr"],
    "vic-a": ["husl1", "husl2", "hlsl"],
}


### READ INPUT DATA ###
#######################

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


### SIGNATURES ###
##################

CSIGN = ["Crc", "Crchf", "Crclf", "Crch2r", "Cfp2", "Cfp10", "Cfp50", "Cfp90"]

ESIGN = ["Eff", "Ebf", "Erc", "Erchf", "Erclf", "Erch2r", "Elt", "Epf"]


### OPTIMIZE ###
################

ALGORITHM = ["sbs", "l-bfgs-b", "nelder-mead"]

JOBS_FUN = [
    "nse",
    "kge",
    "kge2",
    "se",
    "rmse",
    "logarithmic",
]

CSIGN_OPTIM = ["Crc"]

ESIGN_OPTIM = ["Erc", "Elt", "Epf"]

MAPPING = ["uniform", "distributed", "hyper-linear", "hyper-polynomial"]