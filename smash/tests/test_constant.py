from __future__ import annotations

from smash._constant import (
    STRUCTURE_NAME,
    SERR_MU_MAPPING_NAME,
    SERR_SIGMA_MAPPING_NAME,
    RR_PARAMETERS,
    RR_STATES,
    SERR_MU_PARAMETERS,
    SERR_SIGMA_PARAMETERS,
    STRUCTURE_RR_PARAMETERS,
    STRUCTURE_RR_STATES,
    SERR_MU_MAPPING_PARAMETERS,
    SERR_SIGMA_MAPPING_PARAMETERS,
    FEASIBLE_RR_PARAMETERS,
    FEASIBLE_RR_INITIAL_STATES,
    FEASIBLE_SERR_MU_PARAMETERS,
    FEASIBLE_SERR_SIGMA_PARAMETERS,
    DEFAULT_RR_PARAMETERS,
    DEFAULT_RR_INITIAL_STATES,
    DEFAULT_SERR_MU_PARAMETERS,
    DEFAULT_SERR_SIGMA_PARAMETERS,
    DEFAULT_BOUNDS_RR_PARAMETERS,
    DEFAULT_BOUNDS_RR_INITIAL_STATES,
    DEFAULT_BOUNDS_SERR_MU_PARAMETERS,
    DEFAULT_BOUNDS_SERR_SIGMA_PARAMETERS,
    INPUT_DATA_FORMAT,
    RATIO_PET_HOURLY,
    DATASET_NAME,
    PEAK_QUANT,
    MAX_DURATION,
)

import numpy as np


def test_structure_model():
    # % Check structure name
    assert STRUCTURE_NAME == [
        "gr4-lr",
        "gr4-kw",
        "gr5-lr",
        "gr5-kw",
        "loieau-lr",
        "grd-lr",
        "vic3l-lr",
    ]

    # % Check opr parameters
    assert RR_PARAMETERS == [
        "ci",
        "cp",
        "ct",
        "kexc",
        "aexc",
        "ca",
        "cc",
        "kb",
        "b",
        "cusl",
        "cmsl",
        "cbsl",
        "ks",
        "pbc",
        "ds",
        "dsm",
        "ws",
        "llr",
        "akw",
        "bkw",
    ]

    # % Check opr states
    assert RR_STATES == [
        "hi",
        "hp",
        "ht",
        "ha",
        "hc",
        "hcl",
        "husl",
        "hmsl",
        "hbsl",
        "hlr",
    ]

    # % Check structure rr parameter
    assert list(STRUCTURE_RR_PARAMETERS.values()) == [
        ["ci", "cp", "ct", "kexc", "llr"],  # % gr4-lr
        ["ci", "cp", "ct", "kexc", "akw", "bkw"],  # % gr4-kw
        ["ci", "cp", "ct", "kexc", "aexc", "llr"],  # % gr5-lr
        ["ci", "cp", "ct", "kexc", "aexc", "akw", "bkw"],  # % gr5-kw
        ["ca", "cc", "kb", "llr"],  # % loieau-lr
        ["cp", "ct", "llr"],  # % grd-lr
        [
            "b",
            "cusl",
            "cmsl",
            "cbsl",
            "ks",
            "pbc",
            "ds",
            "dsm",
            "ws",
            "llr",
        ],  # % vic3l-lr
    ]

    # % Check structure rr state
    assert list(STRUCTURE_RR_STATES.values()) == [
        ["hi", "hp", "ht", "hlr"],  # % gr4-lr
        ["hi", "hp", "ht"],  # % gr4-kw
        ["hi", "hp", "ht", "hlr"],  # % gr5-lr
        ["hi", "hp", "ht"],  # % gr5-kw
        ["ha", "hc", "hlr"],  # % loieau-lr
        ["hp", "ht", "hlr"],  # % grd-lr
        ["hcl", "husl", "hmsl", "hbsl", "hlr"],  # % vic3l-lr
    ]

    # % Check mu mapping name
    assert SERR_MU_MAPPING_NAME == ["Zero", "Constant", "Linear"]

    # % Check sigma mapping name
    assert SERR_SIGMA_MAPPING_NAME == [
        "Constant",
        "Linear",
        "Power",
        "Exponential",
        "Gaussian",
    ]

    # % Check serr mu parameters
    assert SERR_MU_PARAMETERS == ["mg0", "mg1"]

    # % Check serr mu parameters
    assert SERR_SIGMA_PARAMETERS == ["sg0", "sg1", "sg2"]

    # % Check mu mapping parameters
    assert list(SERR_MU_MAPPING_PARAMETERS.values()) == [
        [],  # % zero
        ["mg0"],  # % constant
        ["mg0", "mg1"],  # % linear
    ]

    # % Check sigma mapping parameters
    assert list(SERR_SIGMA_MAPPING_PARAMETERS.values()) == [
        ["sg0"],  # % constant
        ["sg0", "sg1"],  # % linear
        ["sg0", "sg1", "sg2"],  # % power
        ["sg0", "sg1", "sg2"],  # % exponential
        ["sg0", "sg1", "sg2"],  # % gaussian
    ]


def test_feasible_domain():
    # % Check feasible rr parameters
    assert list(FEASIBLE_RR_PARAMETERS.values()) == [
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
    ]

    # % Check feasible rr states
    assert list(FEASIBLE_RR_INITIAL_STATES.values()) == [
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
    ]

    # % Check feasible serr mu parameters
    assert list(FEASIBLE_SERR_MU_PARAMETERS.values()) == [
        (-np.inf, np.inf),  # % mg0
        (-np.inf, np.inf),  # % mg1
    ]

    # % Check feasible serr sigma parameters
    assert list(FEASIBLE_SERR_SIGMA_PARAMETERS.values()) == [
        (0, np.inf),  # % sg0
        (0, np.inf),  # % sg1
        (0, np.inf),  # % sg2
    ]


def test_default_parameters():
    # % Check default rr parameters
    assert list(DEFAULT_RR_PARAMETERS.values()) == [
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
    ]

    # % Check default rr states
    assert list(DEFAULT_RR_INITIAL_STATES.values()) == [
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
    ]

    # % Check default serr mu parameters
    assert list(DEFAULT_SERR_MU_PARAMETERS.values()) == [
        0,  # % mg0
        0,  # % mg1
    ]

    # % Check default serr sigma parameters
    assert list(DEFAULT_SERR_SIGMA_PARAMETERS.values()) == [
        1,  # % sg0
        0.2,  # % sg1
        2,  # % sg2
    ]


def test_default_bounds_parameters():
    # % Check default bounds rr parameters
    assert list(DEFAULT_BOUNDS_RR_PARAMETERS.values()) == [
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
    ]

    # % Check default bounds rr states
    assert list(DEFAULT_BOUNDS_RR_INITIAL_STATES.values()) == [
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
    ]

    # % Check default bounds serr mu parameters
    assert list(DEFAULT_BOUNDS_SERR_MU_PARAMETERS.values()) == [
        (-1e6, 1e6),  # % mg0
        (-1e6, 1e6),  # % mg1
    ]

    # % Check default bounds serr sigma parameters
    assert list(DEFAULT_BOUNDS_SERR_SIGMA_PARAMETERS.values()) == [
        (1e-6, 1e3),  # % sg0
        (1e-6, 1e1),  # % sg1
        (1e-6, 1e3),  # % sg2
    ]


def test_read_input_data():
    # % Check input data format
    assert INPUT_DATA_FORMAT == ["tif", "nc"]

    # % Check ratio_pet_hourly
    assert np.array_equal(
        RATIO_PET_HOURLY,
        np.array(
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
        ),
    )


def test_event_seg():
    assert PEAK_QUANT == 0.995
    assert MAX_DURATION == 240


def test_dataset_name():
    assert DATASET_NAME == ["flwdir", "cance", "lez", "france"]
