from __future__ import annotations

from smash.fcore._mwd_stats import StatsDT
from smash._constant import (
    INTERNAL_FLUXES
)

import numpy as np

def _get_idx(stats:StatsDT, name):
        for i, key in enumerate(stats.fluxes_keys):
            if key == name:
                return i
    
def get_fluxes(stats:StatsDT, name):
    i = _get_idx(stats, name)
    mean = stats.fluxes_values[:, :, 0, i]
    var = stats.fluxes_values[:, :, 1, i]
    minimum = stats.fluxes_values[:, :, 2, i]
    maximum = stats.fluxes_values[:, :, 3, i]
    median = stats.fluxes_values[:, :, 4, i]
    return mean, var, minimum, maximum, median

'''
def get_states(stats:StatsDT, name):
    for i, key in enumerate()
    return 
'''
