from __future__ import annotations

from smash._constant import (
    STRUCTURE_NAME,
    OPR_PARAMETERS,
    OPR_STATES,
    STRUCTURE_OPR_PARAMETERS,
    STRUCTURE_OPR_STATES,
    STRUCTURE_COMPUTE_CI,
    FEASIBLE_OPR_PARAMETERS,
    FEASIBLE_OPR_INITIAL_STATES,
    DEFAULT_OPR_PARAMETERS,
    DEFAULT_OPR_INITIAL_STATES,
    DEFAULT_BOUNDS_OPR_PARAMETERS,
    DEFAULT_BOUNDS_OPR_INITIAL_STATES,
    OPTIMIZABLE_OPR_PARAMETERS,
    OPTIMIZABLE_OPR_INITIAL_STATES,
    INPUT_DATA_FORMAT,
    RATIO_PET_HOURLY,
    DATASET_NAME,
    PEAK_QUANT,
    MAX_DURATION,
)

import numpy as np


def test_structure_model():
    # % Check structure name
    assert STRUCTURE_NAME == ["gr4-lr", "gr4-kw", "gr5-lr", "gr5-kw", "grd-lr"]

    # % Check opr parameters
    assert OPR_PARAMETERS == ["ci", "cp", "ct", "kexc", "texc", "llr", "akw", "bkw"]

    # % Check opr states
    assert OPR_STATES == ["hi", "hp", "ht", "hlr"]

    # % Check structure opr parameter
    assert list(STRUCTURE_OPR_PARAMETERS.values()) == [
        ["ci", "cp", "ct", "kexc", "llr"],
        ["ci", "cp", "ct", "kexc", "akw", "bkw"],
        ["ci", "cp", "ct", "kexc", "texc", "llr"],
        ["ci", "cp", "ct", "kexc", "texc", "akw", "bkw"],
        ["cp", "ct", "llr"],
    ]

    # % Check structure opr state
    assert list(STRUCTURE_OPR_STATES.values()) == [
        ["hi", "hp", "ht", "hlr"],
        ["hi", "hp", "ht"],
        ["hi", "hp", "ht", "hlr"],
        ["hi", "hp", "ht"],
        ["hp", "ht", "hlr"],
    ]

    # % Check compute ci structure
    assert list(STRUCTURE_COMPUTE_CI.values()) == [True, True, True, True, False]


def test_feasible_domain():
    # % Feasible opr parameters
    assert list(FEASIBLE_OPR_PARAMETERS.values()) == [
        (0, np.inf),
        (0, np.inf),
        (0, np.inf),
        (-np.inf, np.inf),
        (0, 1),
        (0, np.inf),
        (0, np.inf),
        (0, np.inf),
    ]

    # % Feasible opr states
    assert list(FEASIBLE_OPR_INITIAL_STATES.values()) == [
        (0, 1),
        (0, 1),
        (0, 1),
        (0, np.inf),
    ]


def test_default_parameters():
    # % Check default opr parameters
    assert list(DEFAULT_OPR_PARAMETERS.values()) == [1e-6, 200, 500, 0, 0.01, 5, 5, 0.6]

    # % Check default opr states
    assert list(DEFAULT_OPR_INITIAL_STATES.values()) == [1e-2, 1e-2, 1e-2, 1e-6]


def test_default_bounds_parameters():
    # % Check default bounds opr parameters
    assert list(DEFAULT_BOUNDS_OPR_PARAMETERS.values()) == [
        (1e-6, 1e2),
        (1e-6, 1e3),
        (1e-6, 1e3),
        (-50, 50),
        (1e-6, 0.999999),
        (1e-6, 1e3),
        (1e-3, 50),
        (1e-3, 1),
    ]

    # % Check default bounds opr states
    assert list(DEFAULT_BOUNDS_OPR_INITIAL_STATES.values()) == [
        (1e-6, 0.999999),
        (1e-6, 0.999999),
        (1e-6, 0.999999),
        (1e-6, 1e3),
    ]


def test_optimizable_parameters():
    # % Check optimizable opr parameters
    assert list(OPTIMIZABLE_OPR_PARAMETERS.values()) == [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    # % Check optimizable opr initial states
    assert list(OPTIMIZABLE_OPR_INITIAL_STATES.values()) == [True, True, True, True]


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
