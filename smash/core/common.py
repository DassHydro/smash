from __future__ import annotations

import numpy as np

SMASH_PARAMETERS = np.array(
    [
        "ci",
        "cp",
        "beta",
        "cft",
        "cst",
        "alpha",
        "exc",
        "lr",
    ],
    dtype="U",
)

SMASH_DEFAULT_PARAMETERS = np.array(
    [
        1.0,
        200.0,
        1000.0,
        500.0,
        500.0,
        0.9,
        0.0,
        5.0,
    ],
    dtype=np.float32,
)

SMASH_LB_PARAMETERS = np.array(
    [
        1e-6,
        1e-6,
        1e-6,
        1e-6,
        1e-6,
        1e-6,
        -50.0,
        1e-6,
    ],
    dtype=np.float32,
)

SMASH_UB_PARAMETERS = np.array(
    [
        1e2,
        1e3,
        1e3,
        1e3,
        1e4,
        0.999999,
        50.0,
        1e3,
    ],
    dtype=np.float32,
)

SMASH_STATES = np.array(
    [
        "hi",
        "hp",
        "hft",
        "hst",
        "hr",
    ],
    dtype="U",
)

SMASH_DEFAULT_STATES = np.array(
    [
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
    ],
    dtype=np.float32,
)

SMASH_RATIO_PET_HOURLY = np.array(
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

SMASH_CONFIGURATION_DICT = {
    "default_parameters": (SMASH_PARAMETERS, SMASH_DEFAULT_PARAMETERS),
    "default_states": (SMASH_STATES, SMASH_DEFAULT_STATES),
    "lb_parameters": (SMASH_PARAMETERS, SMASH_LB_PARAMETERS),
    "ub_parameters": (SMASH_PARAMETERS, SMASH_UB_PARAMETERS),
    "optim_parameters": (SMASH_PARAMETERS, np.zeros(shape=len(SMASH_PARAMETERS))),
}
