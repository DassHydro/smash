from __future__ import annotations

from smash.fcore._mwd_stats import StatsDT
from smash._constant import INTERNAL_FLUXES

import numpy as np


def _get_idx(stats: StatsDT, name):
    for i, key in enumerate(stats.fluxes_keys):
        if key == name:
            draw = "fluxes"
            return i, draw
    for i, key in enumerate(stats.rr_states_keys):
        if key == name:
            draw = "states"
            return i, draw


def get(stats: StatsDT, name):
    i, draw = _get_idx(stats, name)
    if draw == "fluxes":
        mean = stats.fluxes_values[:, :, 0, i]
        var = stats.fluxes_values[:, :, 1, i]
        minimum = stats.fluxes_values[:, :, 2, i]
        maximum = stats.fluxes_values[:, :, 3, i]
        median = stats.fluxes_values[:, :, 4, i]
    if draw == "states":
        mean = stats.rr_states_values[:, :, 0, i]
        var = stats.rr_states_values[:, :, 1, i]
        minimum = stats.rr_states_values[:, :, 2, i]
        maximum = stats.rr_states_values[:, :, 3, i]
        median = stats.rr_states_values[:, :, 4, i]
    return mean, var, minimum, maximum, median


"""
def get_states(stats:StatsDT, name):
    for i, key in enumerate()
    return 
"""
