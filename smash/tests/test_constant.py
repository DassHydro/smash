from __future__ import annotations

import numpy as np

from smash._constant import (
    DATASET_NAME,
    DEFAULT_BOUNDS_RR_INITIAL_STATES,
    DEFAULT_BOUNDS_RR_PARAMETERS,
    DEFAULT_BOUNDS_SERR_MU_PARAMETERS,
    DEFAULT_BOUNDS_SERR_SIGMA_PARAMETERS,
    DEFAULT_RR_INITIAL_STATES,
    DEFAULT_RR_PARAMETERS,
    DEFAULT_SERR_MU_PARAMETERS,
    DEFAULT_SERR_SIGMA_PARAMETERS,
    FEASIBLE_RR_INITIAL_STATES,
    FEASIBLE_RR_PARAMETERS,
    FEASIBLE_SERR_MU_PARAMETERS,
    FEASIBLE_SERR_SIGMA_PARAMETERS,
    HYDROLOGICAL_MODULE,
    HYDROLOGICAL_MODULE_RR_PARAMETERS,
    HYDROLOGICAL_MODULE_RR_STATES,
    INPUT_DATA_FORMAT,
    MAX_DURATION,
    PEAK_QUANT,
    RATIO_PET_HOURLY,
    ROUTING_MODULE,
    ROUTING_MODULE_NQZ,
    ROUTING_MODULE_RR_PARAMETERS,
    ROUTING_MODULE_RR_STATES,
    RR_PARAMETERS,
    RR_STATES,
    SERR_MU_MAPPING,
    SERR_MU_MAPPING_PARAMETERS,
    SERR_MU_PARAMETERS,
    SERR_SIGMA_MAPPING,
    SERR_SIGMA_MAPPING_PARAMETERS,
    SERR_SIGMA_PARAMETERS,
    SNOW_MODULE,
    SNOW_MODULE_RR_PARAMETERS,
    SNOW_MODULE_RR_STATES,
)


def test_module_name():
    # % Check snow module name
    assert SNOW_MODULE == ["zero", "ssn"]

    # % Check hydrological module
    assert HYDROLOGICAL_MODULE == ["gr4", "gr5", "grd", "loieau", "vic3l"]

    # % Check routing module
    assert ROUTING_MODULE == ["lag0", "lr", "kw"]


def test_module_parameters():
    # % Check snow module rr parameters
    assert list(SNOW_MODULE_RR_PARAMETERS.values()) == [
        [],  # % zero
        ["kmlt"],  # % ssn
    ]

    # % Check snow module rr states
    assert list(SNOW_MODULE_RR_STATES.values()) == [
        [],  # % zero
        ["hs"],  # % ssn
    ]

    # % Check hydrological module rr parameters
    assert list(HYDROLOGICAL_MODULE_RR_PARAMETERS.values()) == [
        ["ci", "cp", "ct", "kexc"],  # % gr4
        ["ci", "cp", "ct", "kexc", "aexc"],  # % gr5
        ["cp", "ct"],  # % grd
        ["ca", "cc", "kb"],  # % loieau
        ["b", "cusl", "cmsl", "cbsl", "ks", "pbc", "ds", "dsm", "ws"],  # % vic3l
    ]

    # % Check hydrological module rr states
    assert list(HYDROLOGICAL_MODULE_RR_STATES.values()) == [
        ["hi", "hp", "ht"],  # % gr4
        ["hi", "hp", "ht"],  # % gr5
        ["hp", "ht"],  # % grd
        ["ha", "hc"],  # % loieau
        ["hcl", "husl", "hmsl", "hbsl"],  # % vic3l
    ]

    # % Check routing module rr parameters
    assert list(ROUTING_MODULE_RR_PARAMETERS.values()) == [[], ["llr"], ["akw", "bkw"]]

    # % Check routing module rr states
    assert list(ROUTING_MODULE_RR_STATES.values()) == [[], ["hlr"], []]

    assert list(ROUTING_MODULE_NQZ.values()) == [1, 1, 2]


def test_parameters():
    # % Check rainfall-runoff parameters
    assert RR_PARAMETERS == [
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

    # % Check rainfall-runoff states
    assert RR_STATES == [
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


def test_structural_error_mapping_name():
    # % Check mu mapping name
    assert SERR_MU_MAPPING == ["Zero", "Constant", "Linear"]

    # % Check sigma mapping name
    assert SERR_SIGMA_MAPPING == [
        "Constant",
        "Linear",
        "Power",
        "Exponential",
        "Gaussian",
    ]


def test_structural_error_mapping_parameters():
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


def test_structural_error_parameters():
    # % Check serr mu parameters
    assert SERR_MU_PARAMETERS == ["mg0", "mg1"]

    # % Check serr sigma parameters
    assert SERR_SIGMA_PARAMETERS == ["sg0", "sg1", "sg2"]


def test_feasible_domain():
    # % Check feasible rr parameters
    assert list(FEASIBLE_RR_PARAMETERS.values()) == [
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
    ]

    # % Check feasible rr states
    assert list(FEASIBLE_RR_INITIAL_STATES.values()) == [
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
    ]

    # % Check default rr states
    assert list(DEFAULT_RR_INITIAL_STATES.values()) == [
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
    ]

    # % Check default bounds rr states
    assert list(DEFAULT_BOUNDS_RR_INITIAL_STATES.values()) == [
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
