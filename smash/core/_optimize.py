from __future__ import annotations

from smash.solver._mwd_common import name_parameters, name_states
from smash.solver._mwd_setup import SetupDT
from smash.solver._mw_optimize import optimize_sbs, optimize_lbfgsb

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model
    from smash.solver._mwd_mesh import MeshDT
    from smash.solver._mwd_input_data import Input_DataDT
    from smash.solver._mwd_parameters import ParametersDT
    from smash.solver._mwd_states import StatesDT
    from smash.solver._mwd_output import OutputDT

import warnings
import numpy as np
import pandas as pd

ALGORITHM = ["sbs", "l-bfgs-b"]

JOBS_FUN = [
    "nse",
    "kge",
    "kge2",
    "se",
    "rmse",
    "logarithmique",
]


def _optimize_sbs(
    instance: Model,
    control_vector: list[str],
    jobs_fun: str,
    bounds: list[float],
    wgauge: list[float],
    ost: pd.Timestamp,
    maxiter: int = 100,
    **unknown_options,
):

    _check_unknown_options(unknown_options)

    setup_bgd = SetupDT()
    
    instance.setup.optim_parameters = 0
    instance.setup.lb_parameters = setup_bgd.lb_parameters.copy()
    instance.setup.ub_parameters = setup_bgd.ub_parameters.copy()

    for i, name in enumerate(control_vector):

        ind = np.argwhere(name_parameters == name)

        instance.setup.optim_parameters[ind] = 1

        instance.setup.lb_parameters[ind] = bounds[i][0]
        instance.setup.ub_parameters[ind] = bounds[i][1]

    st = pd.Timestamp(instance.setup.start_time.decode().strip())

    instance.setup.optim_start_step = (ost - st).total_seconds() / instance.setup.dt + 1

    instance.setup.algorithm = "sbs"
    instance.setup.jobs_fun = jobs_fun
    instance.setup.maxiter = maxiter
    
    instance.mesh.wgauge = wgauge

    optimize_sbs(
        instance.setup,
        instance.mesh,
        instance.input_data,
        instance.parameters,
        instance.states,
        instance.output,
    )


def _optimize_lbfgsb(
    instance: Model,
    control_vector: list[str],
    jobs_fun: str,
    bounds: list[float],
    wgauge: list[float],
    ost: pd.Timestamp,
    maxiter: int = 100,
    **unknown_options,
):

    _check_unknown_options(unknown_options)

    setup_bgd = SetupDT()

    instance.setup.optim_parameters = 0
    instance.setup.optim_states = 0
    instance.setup.lb_parameters = setup_bgd.lb_parameters.copy()
    instance.setup.lb_states = setup_bgd.lb_states.copy()
    instance.setup.ub_states = setup_bgd.ub_states.copy()

    for i, name in enumerate(control_vector):

        if name in name_parameters:

            ind = np.argwhere(name_parameters == name)

            instance.setup.optim_parameters[ind] = 1

            instance.setup.lb_parameters[ind] = bounds[i][0]
            instance.setup.ub_parameters[ind] = bounds[i][1]

        #% Already check, must be states if not parameters
        else:

            ind = np.argwhere(name_states == name)

            instance.setup.optim_states[ind] = 1

            instance.setup.lb_states[ind] = bounds[i][0]
            instance.setup.ub_states[ind] = bounds[i][1]

    st = pd.Timestamp(instance.setup.start_time.decode().strip())

    instance.setup.optim_start_step = (ost - st).total_seconds() / instance.setup.dt + 1

    instance.setup.algorithm = "l-bfgs-b"
    instance.setup.jobs_fun = jobs_fun
    instance.setup.maxiter = maxiter
    
    instance.mesh.wgauge = wgauge

    optimize_lbfgsb(
        instance.setup,
        instance.mesh,
        instance.input_data,
        instance.parameters,
        instance.states,
        instance.output,
    )


def _standardize_algorithm(algorithm) -> str:

    if isinstance(algorithm, str):

        if algorithm.lower() in ALGORITHM:

            algorithm = algorithm.lower()

        else:

            raise ValueError(f"Unknown algorithm '{algorithm}'")

    else:

        raise TypeError(f"'algorithm' argument must be str")

    return algorithm


def _standardize_control_vector(control_vector, algorithm) -> list:

    if isinstance(control_vector, (list, tuple, set)):

        control_vector = list(control_vector)

        for name in control_vector:

            if not name in [*name_parameters, *name_states]:

                raise ValueError(
                    f"Unknown parameter or state '{name}' in 'control_vector'"
                )

            elif name in name_states and algorithm != "l-bfgs-b":

                raise ValueError(
                    f"state optimization not available with '{algorithm}' algorithm"
                )
    else:

        raise TypeError(f"'control_vector' argument must be list-like object")

    return control_vector


def _standardize_jobs_fun(jobs_fun, algorithm) -> str:

    if isinstance(jobs_fun, str):

        if not jobs_fun in JOBS_FUN:

            raise ValueError(f"Unknown objective function ('jobs_fun') '{jobs_fun}'")

        elif jobs_fun in ["kge"] and algorithm == "l-bfgs-b":

            raise ValueError(
                f"'{jobs_fun}' objective function can not be use with 'l-bfgs-b' algorithm (non convex function)"
            )

    else:

        raise TypeError(f"'jobs_fun' argument must be str")

    return jobs_fun


def _standardize_bounds(bounds, control_vector, setup) -> list:

    if bounds:

        if isinstance(bounds, (list, tuple, set)):

            bounds = list(bounds)

            if len(bounds) != len(control_vector):

                raise ValueError(
                    f"Inconsistent size between 'control_vector' ({len(control_vector)}) and 'bounds' ({len(bounds)})"
                )

            else:

                for i, lb_ub in enumerate(bounds):

                    if not isinstance(lb_ub, (list, tuple, set)) or len(lb_ub) != 2:

                        raise ValueError(
                            f"'bounds' values for '{control_vector[i]}' must be list-like object of length 2"
                        )

                    if lb_ub[0] > lb_ub[1]:

                        raise ValueError(
                            f"'bounds' values for '{control_vector[i]}' is invalid lower bound ({lb_ub[0]}) greater than upper bound ({lb_ub[1]})"
                        )
        else:

            raise TypeError(f"'bounds' argument must be list-like object")

    else:

        bounds = []

        for name in control_vector:

            if name in name_parameters:

                ind = np.argwhere(name_parameters == name)

                bounds.append(
                    (setup.lb_parameters[ind].item(), setup.ub_parameters[ind].item())
                )

            elif name in name_states:

                ind = np.argwhere(name_states == name)

                bounds.append(
                    (setup.lb_states[ind].item(), setup.ub_states[ind].item())
                )

    return bounds


def _standardize_gauge(gauge, setup, mesh, input_data) -> list:

    code = np.array(mesh.code.tobytes(order="F").decode().split())

    if gauge:

        if isinstance(gauge, (list, tuple, set)):

            gauge_imd = gauge.copy()

            gauge = list(gauge)

            for i, name in enumerate(gauge):

                if name in code:

                    ind = np.argwhere(code == name)

                    if np.all(input_data.qobs[ind, setup.optim_start_step :] < 0):

                        gauge_imd.remove(name)

                        warnings.warn(
                            f"gauge '{name}' has no available observed discharge. Removed from the optimization"
                        )

                else:

                    raise ValueError(
                        f"gauge code '{name}' does not belong to the list of 'Model' gauges code {code}"
                    )

            if len(gauge_imd) == 0:

                raise ValueError(
                    f"No available observed discharge for optimization at gauge {gauge}"
                )

            else:

                gauge = gauge_imd.copy()

        elif isinstance(gauge, str):

            if gauge == "all":

                gauge = code.copy()

                gauge_imd = gauge.copy()

                for ind, name in enumerate(code):

                    if np.all(input_data.qobs[ind, setup.optim_start_step :] < 0):

                        gauge_imd.remove(name)

                        warnings.warn(
                            f"gauge '{name}' has no available observed discharge. Removed from the optimization"
                        )

                if len(gauge_imd) == 0:

                    raise ValueError(
                        f"No available observed discharge for optimization at gauge {gauge}"
                    )

                else:

                    gauge = gauge_imd.copy()

            elif gauge == "downstream":

                ind = np.argmax(mesh.area)

                if np.all(input_data.qobs[ind, setup.optim_start_step :] < 0):

                    raise ValueError(
                        f"No available observed discharge for optimization at gauge {gauge}"
                    )

                else:

                    gauge = [code[ind]]

            elif gauge in code:

                ind = np.argwhere(code == gauge)

                if np.all(input_data.qobs[ind, setup.optim_start_step :] < 0):

                    raise ValueError(
                        f"No available observed discharge for optimization at gauge {gauge}"
                    )

                else:

                    gauge = [gauge]

            else:

                raise ValueError(
                    f"str 'gauge' argument must be either one alias among ['all', 'downstream'] or a gauge code {code}"
                )

        else:
            raise TypeError(f"'gauge' argument must be list-like object or str")

    else:

        ind = np.argmax(mesh.area)

        if np.all(input_data.qobs[ind, setup.optim_start_step :] < 0):

            raise ValueError(
                f"No available observed discharge for optimization at gauge {code[ind]}"
            )

        else:

            gauge = [code[ind]]

    return gauge


def _standardize_wgauge(wgauge, gauge, mesh) -> list:
    
    code = np.array(mesh.code.tobytes(order="F").decode().split())

    if wgauge:

        if isinstance(wgauge, (list, tuple, set)):
            
            imd_wgauge = list(wgauge)
            
            if len(imd_wgauge) != len(gauge):

                raise ValueError(
                    f"Inconsistent size between 'gauge' ({len(gauge)}) and 'wgauge' ({len(imd_wgauge)})"
                )
                
            else:
                
                wgauge = np.zeros(shape=len(code), dtype=np.float32)
                
                ind = np.in1d(code, gauge)
                
                wgauge[ind] = imd_wgauge
                
        elif isinstance(wgauge, (int, float)):
            
            imd_wgauge = [wgauge]
            
            if len(imd_wgauge) != len(gauge):

                raise ValueError(
                    f"Inconsistent size between 'gauge' ({len(gauge)}) and 'wgauge' ({len(imd_wgauge)})"
                )
                
            else:
                
                wgauge = np.zeros(shape=len(code), dtype=np.float32)
                
                ind = np.in1d(code, gauge)
                
                wgauge[ind] = imd_wgauge
                
        
        elif isinstance(wgauge, str):
            
            if wgauge == "mean":
                
                wgauge = np.zeros(shape=len(code), dtype=np.float32)
                
                wgauge = np.where(np.in1d(code, gauge), 1/len(gauge), wgauge)
                
            elif wgauge == "area":
                
                wgauge = np.zeros(shape=len(code), dtype=np.float32)
                
                ind = np.in1d(code, gauge)
                
                value = mesh.area[ind] / sum(mesh.area[ind])
                
                wgauge[ind] = value
                
            elif wgauge == "area_minv":
                
                wgauge = np.zeros(shape=len(code), dtype=np.float32)
                
                ind = np.in1d(code, gauge)
                
                value = (1 / mesh.area[ind]) / sum(1 / mesh.area[ind])
                
                wgauge[ind] = value
                
            else:
                
                raise ValueError(
                    f"str 'wgauge' argument must be either one alias among ['mean', 'area', 'area_minv']"
                )
                
        else:
            
            raise TypeError(f"'wgauge' argument must be list-like object, int, float or str")
            
    else:
        
        #% Default is mean
        wgauge = np.zeros(shape=len(code), dtype=np.float32)
        
        wgauge = np.where(np.in1d(code, gauge), 1/len(gauge), wgauge)

    return wgauge


def _standardize_ost(ost, setup) -> pd.Timestamp:

    st = pd.Timestamp(setup.start_time.decode().strip())
    et = pd.Timestamp(setup.end_time.decode().strip())

    if ost:

        try:
            ost = pd.Timestamp(ost)

            if (ost - st).total_seconds() < 0 or (et - ost).total_seconds() < 0:

                raise ValueError(
                    f"'ost' ({ost}) argument must be between start time ({st}) and end time ({et})"
                )

        except:

            raise ValueError(f"'ost' ({ost}) argument is an invalid date")

    else:

        ost = st

    return ost


def _check_unknown_options(unknown_options):

    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        warnings.warn("Unknown algorithm options: %s" % msg)


def _standardize_optimize_args(
    algorithm,
    control_vector,
    jobs_fun,
    bounds,
    gauge,
    wgauge,
    ost,
    setup,
    mesh,
    input_data,
):

    algorithm = _standardize_algorithm(algorithm)

    control_vector = _standardize_control_vector(control_vector, algorithm)

    jobs_fun = _standardize_jobs_fun(jobs_fun, algorithm)

    bounds = _standardize_bounds(bounds, control_vector, setup)

    gauge = _standardize_gauge(gauge, setup, mesh, input_data)

    wgauge = _standardize_wgauge(wgauge, gauge, mesh)

    ost = _standardize_ost(ost, setup)

    return algorithm, control_vector, jobs_fun, bounds, wgauge, ost
