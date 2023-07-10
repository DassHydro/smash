from __future__ import annotations

from smash._constant import (
    STRUCTURE_OPR_PARAMETERS,
    STRUCTURE_OPR_STATES,
    BOUNDS_OPR_PARAMETERS,
    BOUNDS_OPR_STATES,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import Timestamp
    from smash.solver._mwd_setup import SetupDT


def _map_dict_to_object(dct: dict, obj: object):
    for key, value in dct.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def _index_containing_substring(the_list: list, substring: str):
    for i, s in enumerate(the_list):
        if substring in s:
            return i
    return -1


def _adjust_left_files_by_date(files: list[str], date_range: Timestamp):
    n = 0
    ind = -1
    while ind == -1:
        ind = _index_containing_substring(files, date_range[n].strftime("%Y%m%d%H%M"))

        n += 1

    return files[ind:]


def _check_unknown_options(type_check: str, unknown_options: dict):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))

        raise KeyError(f"Unknown {type_check} options: '{msg}'")


def _default_bound_constraints(setup: SetupDT, states: bool):
    if states:
        control_vector = STRUCTURE_OPR_STATES[setup.structure]

    else:
        control_vector = STRUCTURE_OPR_PARAMETERS[setup.structure]

    bounds = []

    for name in control_vector:
        if name in BOUNDS_OPR_STATES:
            l, u = BOUNDS_OPR_STATES[name]

        else:
            l, u = BOUNDS_OPR_PARAMETERS[name]

        bounds += [[l, u]]

    problem = {
        "num_vars": len(control_vector),
        "names": control_vector,
        "bounds": bounds,
    }

    return problem
