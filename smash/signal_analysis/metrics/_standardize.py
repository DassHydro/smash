from __future__ import annotations

from smash._constant import EFFICIENCY_METRICS

import numpy as np


def _standardize_arrays(obs: np.ndarray, sim: np.ndarray):

    if isinstance(obs, np.ndarray) and isinstance(sim, np.ndarray):

        if obs.ndim <= 2 and sim.ndim <= 2 and obs.shape == sim.shape:

            if obs.ndim == 1:
                obs = obs[np.newaxis, ...]
                sim = sim[np.newaxis, ...]
            
            return obs, sim
        
        else:
            raise ValueError("obs and sim must be 1D- or 2D-arrays with the same shape")
    
    else:
        raise TypeError("obs and sim must be NumPy arrays")
    

def _standardize_metric(metric: str):

    if isinstance(metric, str):
        if metric in EFFICIENCY_METRICS:

            return metric.lower()
        
        else:
            raise ValueError(f"Unknown efficiency metric {metric}. Choices: {EFFICIENCY_METRICS}")
    else:
        raise TypeError(f"metric must be str")