from __future__ import annotations

import os
import platform
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from smash._constant import (
    ADAPTIVE_OPTIMIZER,
    DEFAULT_SIMULATION_COMMON_OPTIONS,
    DEFAULT_SIMULATION_COST_OPTIONS,
    DEFAULT_SIMULATION_RETURN_OPTIONS,
    DEFAULT_TERMINATION_CRIT,
    EVENT_SEG_KEYS,
    F_PRECISION,
    FEASIBLE_RR_INITIAL_STATES,
    FEASIBLE_RR_PARAMETERS,
    FEASIBLE_SERR_MU_PARAMETERS,
    FEASIBLE_SERR_SIGMA_PARAMETERS,
    GAUGE_ALIAS,
    JOBS_CMPT,
    JOBS_CMPT_TFM,
    JREG_CMPT,
    MAPPING,
    MAPPING_OPTIMIZER,
    METRICS,
    NN_PARAMETERS_KEYS,
    OPTIMIZABLE_NN_PARAMETERS,
    OPTIMIZABLE_RR_INITIAL_STATES,
    OPTIMIZABLE_RR_PARAMETERS,
    OPTIMIZABLE_SERR_MU_PARAMETERS,
    OPTIMIZABLE_SERR_SIGMA_PARAMETERS,
    OPTIMIZER_CLASS,
    OPTIMIZER_CONTROL_TFM,
    REGIONAL_MAPPING,
    RR_PARAMETERS,
    RR_STATES,
    SERR_MU_MAPPING_PARAMETERS,
    SERR_SIGMA_MAPPING_PARAMETERS,
    SIGNS,
    SIMULATION_OPTIMIZE_OPTIONS_KEYS,
    STRUCTURE_RR_PARAMETERS,
    STRUCTURE_RR_STATES,
    WEIGHT_ALIAS,
    WJREG_ALIAS,
)

# Used inside eval statement
from smash.core.signal_analysis.segmentation._standardize import (  # noqa: F401
    _standardize_hydrograph_segmentation_max_duration,
    _standardize_hydrograph_segmentation_peak_quant,
)
from smash.core.signal_analysis.segmentation._tools import _mask_event

# Used inside eval statement
from smash.factory.net._optimizers import SGD, Adagrad, Adam, RMSprop  # noqa: F401
from smash.factory.net.net import Net
from smash.factory.samples.samples import Samples

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.util._typing import AlphaNumeric, AnyTuple, ListLike, Numeric


def _standardize_simulation_mapping(mapping: str) -> str:
    if isinstance(mapping, str):
        if mapping.lower() not in MAPPING:
            raise ValueError(f"Unknown mapping '{mapping}'. Choices: {MAPPING}")
    else:
        raise TypeError("mapping argument must be a str")

    return mapping.lower()


def _standardize_simulation_optimizer(mapping: str, optimizer: str | None) -> str:
    if optimizer is None:
        optimizer = MAPPING_OPTIMIZER[mapping][0]

    else:
        if isinstance(optimizer, str):
            if optimizer.lower() not in MAPPING_OPTIMIZER[mapping]:
                raise ValueError(
                    f"Unknown optimizer '{optimizer}' for mapping '{mapping}'. Choices: "
                    f"{MAPPING_OPTIMIZER[mapping]}"
                )

        else:
            raise TypeError("optimizer argument must be a str")

    return optimizer.lower()


def _standardize_simulation_samples(model: Model, samples: Samples) -> Samples:
    if isinstance(samples, Samples):
        for key in samples._problem["names"]:
            if key in model.rr_parameters.keys:
                low, upp = FEASIBLE_RR_PARAMETERS[key]
            elif key in model.rr_initial_states.keys:
                low, upp = FEASIBLE_RR_INITIAL_STATES[key]
            else:
                available_parameters = list(model.rr_parameters.keys) + list(model.rr_initial_states.keys)
                raise ValueError(
                    f"Unknown parameter '{key}' in samples attributes. Choices: {available_parameters}"
                )

            # % Check that sample is inside feasible domain
            arr = getattr(samples, key)
            low_arr = np.min(arr)
            upp_arr = np.max(arr)
            if (low_arr + F_PRECISION) <= low or (upp_arr - F_PRECISION) >= upp:
                raise ValueError(
                    f"Invalid sample values for parameter '{key}'. Sample domain [{low_arr}, {upp_arr}] is "
                    f"not included in the feasible domain ]{low}, {upp}["
                )
    else:
        raise TypeError("samples arguments must be a smash.Samples object")

    return samples


def _standardize_simulation_optimize_options_parameters(
    model: Model, func_name: str, parameters: str | ListLike | None, **kwargs
) -> np.ndarray:
    is_bayesian = "bayesian" in func_name

    available_rr_parameters = [
        key for key in STRUCTURE_RR_PARAMETERS[model.setup.structure] if OPTIMIZABLE_RR_PARAMETERS[key]
    ]
    available_rr_initial_states = [
        key for key in STRUCTURE_RR_STATES[model.setup.structure] if OPTIMIZABLE_RR_INITIAL_STATES[key]
    ]
    available_serr_mu_parameters = [
        key
        for key in SERR_MU_MAPPING_PARAMETERS[model.setup.serr_mu_mapping]
        if OPTIMIZABLE_SERR_MU_PARAMETERS[key]
    ]
    available_serr_sigma_parameters = [
        key
        for key in SERR_SIGMA_MAPPING_PARAMETERS[model.setup.serr_sigma_mapping]
        if OPTIMIZABLE_SERR_SIGMA_PARAMETERS[key]
    ]
    available_nn_parameters = OPTIMIZABLE_NN_PARAMETERS[max(0, model.setup.n_layers - 1)]

    available_parameters = available_rr_parameters + available_rr_initial_states + available_nn_parameters

    if is_bayesian:
        available_parameters.extend(available_serr_mu_parameters + available_serr_sigma_parameters)

    if parameters is None:
        default_parameters = available_rr_parameters + available_nn_parameters

        if is_bayesian:
            default_parameters += available_serr_mu_parameters + available_serr_sigma_parameters

        parameters = np.array(default_parameters, ndmin=1)

    else:
        if isinstance(parameters, (str, list, tuple, np.ndarray)):
            parameters = np.array(parameters, ndmin=1)

            for i, prmt in enumerate(parameters):
                if prmt in available_parameters:
                    parameters[i] = prmt.lower()

                else:
                    raise ValueError(
                        f"Unknown or non optimizable parameter '{prmt}' at index {i} in parameters "
                        f"optimize_options. Choices: {available_parameters}"
                    )
        else:
            raise TypeError(
                "parameters optimize_options must be a str or ListLike type (List, Tuple, np.ndarray)"
            )

    # % Sort parameters
    parameters = [prmt for prmt in available_parameters if prmt in parameters]

    return parameters


def _standardize_simulation_optimize_options_bounds(
    model: Model, parameters: np.ndarray, bounds: dict | None, **kwargs
) -> dict:
    bounded_parameters = [p for p in parameters if p not in NN_PARAMETERS_KEYS]

    if bounds is None:
        bounds = {}

    else:
        if isinstance(bounds, dict):
            for key, value in bounds.items():
                if key in bounded_parameters:
                    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2:
                        if value[0] >= value[1]:
                            raise ValueError(
                                f"Lower bound value {value[0]} is greater than or equal to upper bound "
                                f"{value[1]} for parameter '{key}' in bounds optimize_options"
                            )
                        else:
                            bounds[key] = tuple(value)
                    else:
                        raise TypeError(
                            f"Bounds values for parameter '{key}' must be of ListLike type (List, Tuple, "
                            f"np.ndarray) of size 2 in bounds optimize_options"
                        )
                else:
                    raise ValueError(
                        f"Unknown, non optimized, or unbounded parameter '{key}' in bounds optimize_options. "
                        f"Choices: {bounded_parameters}"
                    )
        else:
            raise TypeError("bounds optimize_options must be a dictionary")

    parameters_bounds = dict(
        **model.get_rr_parameters_bounds(),
        **model.get_rr_initial_states_bounds(),
        **model.get_serr_mu_parameters_bounds(),
        **model.get_serr_sigma_parameters_bounds(),
    )

    for key, value in parameters_bounds.items():
        if key in bounded_parameters:
            bounds.setdefault(key, value)

    # % Check that bounds are inside feasible domain and that bounds include parameter domain
    for key, value in bounds.items():
        if key in model.rr_parameters.keys:
            arr = model.get_rr_parameters(key)
            low, upp = FEASIBLE_RR_PARAMETERS[key]
            # Do not check if a value is inside the feasible domain outside of active cells
            mask = model.mesh.active_cell == 1
        elif key in model.rr_initial_states.keys:
            arr = model.get_rr_initial_states(key)
            low, upp = FEASIBLE_RR_INITIAL_STATES[key]
            # Do not check if a value is inside the feasible domain outside of active cells
            mask = model.mesh.active_cell == 1
        elif key in model.serr_sigma_parameters.keys:
            arr = model.get_serr_sigma_parameters(key)
            low, upp = FEASIBLE_SERR_SIGMA_PARAMETERS[key]
            # Check all values
            mask = np.ones(arr.shape, dtype=bool)
        elif key in model.serr_mu_parameters.keys:
            arr = model.get_serr_mu_parameters(key)
            low, upp = FEASIBLE_SERR_MU_PARAMETERS[key]
            # Check all values
            mask = np.ones(arr.shape, dtype=bool)
        # % In case we have other kind of parameters. Should be unreachable.
        else:
            pass

        low_arr = np.min(arr, where=mask, initial=np.inf)
        upp_arr = np.max(arr, where=mask, initial=-np.inf)
        if (low_arr + F_PRECISION) < value[0] or (upp_arr - F_PRECISION) > value[1]:
            raise ValueError(
                f"Invalid bounds values for parameter '{key}'. Bounds domain [{value[0]}, {value[1]}] does "
                f"not include parameter domain [{low_arr}, {upp_arr}] in bounds optimize_options"
            )

        if (value[0] + F_PRECISION) <= low or (value[1] - F_PRECISION) >= upp:
            raise ValueError(
                f"Invalid bounds values for parameter '{key}'. Bounds domain [{value[0]}, {value[1]}] is not "
                f"included in the feasible domain ]{low}, {upp}[ in bounds optimize_options"
            )

    bounds = {key: bounds[key] for key in bounded_parameters}

    return bounds


def _standardize_simulation_optimize_options_control_tfm(
    mapping: str, optimizer: str, control_tfm: str | None, **kwargs
) -> str | None:
    if control_tfm is None:
        control_tfm = OPTIMIZER_CONTROL_TFM[(mapping, optimizer)][0]
    else:
        if isinstance(control_tfm, str):
            if control_tfm.lower() not in OPTIMIZER_CONTROL_TFM[(mapping, optimizer)]:
                raise ValueError(
                    f"Unknown transformation '{control_tfm}' in control_tfm optimize_options. "
                    f"Choices: {OPTIMIZER_CONTROL_TFM[(mapping, optimizer)]}"
                )
        else:
            raise TypeError("control_tfm optimize_options must be a str")

    return control_tfm


def _standardize_simulation_optimize_options_descriptor(
    model: Model, parameters: np.ndarray, descriptor: dict | None, **kwargs
) -> dict:
    desc_linked_parameters = [key for key in parameters if key in RR_PARAMETERS + RR_STATES]
    if descriptor is None:
        descriptor = {}

    else:
        if isinstance(descriptor, dict):
            for key, value in descriptor.items():
                if key in desc_linked_parameters:
                    if isinstance(value, (str, list, tuple, np.ndarray)):
                        value = np.array(value, ndmin=1)
                        for vle in value:
                            if vle not in model.setup.descriptor_name:
                                raise ValueError(
                                    f"Unknown descriptor '{vle}' for parameter '{key}' in descriptor "
                                    f"optimize_options. Choices: {list(model.setup.descriptor_name)}"
                                )
                        descriptor[key] = value
                    else:
                        raise TypeError(
                            f"Descriptor values for parameter '{key}' must be a str or ListLike type (List, "
                            f"Tuple, np.ndarray) in descriptor optimize_options"
                        )
                else:
                    raise ValueError(
                        f"Unknown or non optimized or non descriptor linked parameter '{key}' in descriptor "
                        f"optimize_options. Choices: {desc_linked_parameters}"
                    )
        else:
            raise TypeError("descriptor optimize_options must be a dictionary")

    for prmt in desc_linked_parameters:
        descriptor.setdefault(prmt, model.setup.descriptor_name)

    return descriptor


def _standardize_simulation_optimize_options_net(
    model: Model, bounds: dict, net: Net | None, **kwargs
) -> Net:
    nrow, ncol, nd = model.physio_data.descriptor.shape

    bound_values = list(bounds.values())
    ncv = len(bound_values)

    if net is None:
        # % Set default graph
        net = Net()

        net.add_dense(nd * 3, input_shape=nd, activation="relu")
        net.add_dense(round(np.sqrt(nd * ncv) * np.log(nrow * ncol)), activation="relu")
        net.add_dense(ncv * 3, activation="relu")
        net.add_dense(ncv, activation="tanh")
        net.add_scale(bound_values)

    elif isinstance(net, Net):
        if net.layers:
            # % Check input shape
            net_in = net.layers[0].input_shape

            x_in = (nrow, ncol, nd) if len(net_in) == 3 else (nd,)  # in case of cnn and mlp resp.

            if net_in != x_in:
                raise ValueError(
                    f"net optimize_options: Inconsistent shapes between the input layer ({net_in}) "
                    f"and the input data ({x_in}): {net_in} != {x_in}"
                )

            # % Check output shape
            net_out = net.layers[-1].output_shape()

            if net_out[-1] != ncv:
                raise ValueError(
                    f"net optimize_options: Inconsistent values between the number of output features "
                    f"({net_out[-1]}) and the number of parameters ({ncv}): {net_out[-1]} != {ncv}"
                )

            # % Check bounds constraints
            if hasattr(net.layers[-1], "_scale_func"):
                net_bounds = np.transpose(
                    [net.layers[-1]._scale_func.lower, net.layers[-1]._scale_func.upper]
                )

                diff = np.not_equal(net_bounds, bound_values)

                for i, name in enumerate(bounds):
                    if diff[i].any():
                        warnings.warn(
                            f"net optimize_options: Inconsistent values between the bounds in scaling layer "
                            f"and the parameter bound for {name}: {net_bounds[i]} != {bound_values[i]}. "
                            f"Ignoring default bounds for scaling layer.",
                            stacklevel=2,
                        )

        else:
            raise ValueError("net optimize_options: The graph has not been set yet")

    else:
        raise TypeError("net optimize_options must be a smash.factory.Net object")

    return net


def _standardize_simulation_optimize_options_learning_rate(
    optimizer: str, learning_rate: Numeric | None, **kwargs
) -> float:
    if isinstance(learning_rate, (int, float)):
        learning_rate = float(learning_rate)
        if learning_rate < 0:
            raise ValueError("learning_rate optimize_options must be greater than 0")
    else:
        if learning_rate is None:
            opt_class = eval(OPTIMIZER_CLASS[ADAPTIVE_OPTIMIZER.index(optimizer)])
            learning_rate = opt_class().learning_rate

        else:
            raise TypeError("learning_rate optimize_options must be of Numeric type (int, float) or None")

    return learning_rate


def _standardize_simulation_optimize_options_random_state(random_state: Numeric | None, **kwargs) -> int:
    if random_state is None:
        pass

    else:
        if not isinstance(random_state, (int, float)):
            raise TypeError("random_state optimize_options must be of Numeric type (int, float)")

        random_state = int(random_state)

        if random_state < 0 or random_state > 4_294_967_295:
            raise ValueError("random_state optimize_options must be between 0 and 2**32 - 1")

    return random_state


def _standardize_simulation_optimize_options_termination_crit(
    optimizer: str, termination_crit: dict | None, **kwargs
) -> dict:
    if termination_crit is None:
        termination_crit = DEFAULT_TERMINATION_CRIT[optimizer].copy()

    else:
        if isinstance(termination_crit, dict):
            pop_keys = []
            for key in termination_crit:
                if key not in DEFAULT_TERMINATION_CRIT[optimizer]:
                    pop_keys.append(key)
                    warnings.warn(
                        f"Unknown termination_crit key '{key}' for optimizer '{optimizer}'. "
                        f"Choices: {list(DEFAULT_TERMINATION_CRIT[optimizer].keys())}",
                        stacklevel=2,
                    )

            for key in pop_keys:
                termination_crit.pop(key)

        else:
            raise TypeError("termination_crit argument must be a dictionary")

    for key, value in DEFAULT_TERMINATION_CRIT[optimizer].items():
        termination_crit.setdefault(key, value)
        func = eval(f"_standardize_simulation_optimize_options_termination_crit_{key}")
        termination_crit[key] = func(optimizer=optimizer, **termination_crit)

    return termination_crit


def _standardize_simulation_optimize_options_termination_crit_maxiter(maxiter: Numeric, **kwargs) -> int:
    if isinstance(maxiter, (int, float)):
        maxiter = int(maxiter)
        if maxiter < 0:
            raise ValueError("maxiter termination_crit must be greater than or equal to 0")
    else:
        raise TypeError("maxiter termination_crit must be of Numeric type (int, float)")

    return maxiter


def _standardize_simulation_optimize_options_termination_crit_xatol(xatol: Numeric, **kwargs) -> float:
    if isinstance(xatol, (int, float)):
        xatol = float(xatol)
        if xatol <= 0:
            raise ValueError("xatol termination_crit must be greater than 0")
    else:
        raise TypeError("xatol termination_crit must be of Numeric type (int, float)")

    return xatol


def _standardize_simulation_optimize_options_termination_crit_fatol(fatol: Numeric, **kwargs) -> float:
    if isinstance(fatol, (int, float)):
        fatol = float(fatol)
        if fatol <= 0:
            raise ValueError("fatol termination_crit must be greater than 0")
    else:
        raise TypeError("fatol termination_crit must be of Numeric type (int, float)")

    return fatol


def _standardize_simulation_optimize_options_termination_crit_factr(factr: Numeric, **kwargs) -> float:
    if isinstance(factr, (int, float)):
        factr = float(factr)
        if factr <= 0:
            raise ValueError("factr termination_crit must be greater than 0")
    else:
        raise TypeError("factr termination_crit must be of Numeric type (int, float)")

    return factr


def _standardize_simulation_optimize_options_termination_crit_pgtol(pgtol: Numeric, **kwargs) -> float:
    if isinstance(pgtol, (int, float)):
        pgtol = float(pgtol)
        if pgtol <= 0:
            raise ValueError("pgtol termination_crit must be greater than 0")
    else:
        raise TypeError("pgtol termination_crit must be of Numeric type (int, float)")

    return pgtol


def _standardize_simulation_optimize_options_termination_crit_early_stopping(
    early_stopping: Numeric, **kwargs
) -> int:
    if isinstance(early_stopping, (int, float)):
        early_stopping = int(early_stopping)

        if early_stopping < 0:
            raise ValueError("early_stopping termination_crit must be non-negative")
    else:
        raise TypeError("early_stopping termination_crit must be of Numeric type (int, float)")

    return early_stopping


def _standardize_simulation_optimize_options(
    model: Model,
    func_name: str,
    mapping: str,
    optimizer: str,
    optimize_options: dict | None,
) -> dict:
    if optimize_options is None:
        optimize_options = dict.fromkeys(SIMULATION_OPTIMIZE_OPTIONS_KEYS[(mapping, optimizer)], None)

    else:
        if isinstance(optimize_options, dict):
            pop_keys = []
            for key in optimize_options:
                if key not in SIMULATION_OPTIMIZE_OPTIONS_KEYS[(mapping, optimizer)]:
                    pop_keys.append(key)
                    warnings.warn(
                        f"Unknown optimize_options key '{key}' for mapping '{mapping}' and optimizer "
                        f"'{optimizer}'. Choices: {SIMULATION_OPTIMIZE_OPTIONS_KEYS[(mapping, optimizer)]}",
                        stacklevel=2,
                    )

            for key in pop_keys:
                optimize_options.pop(key)

        else:
            raise TypeError("optimize_options argument must be a dictionary")

    for key in SIMULATION_OPTIMIZE_OPTIONS_KEYS[(mapping, optimizer)]:
        optimize_options.setdefault(key, None)
        func = eval(f"_standardize_simulation_optimize_options_{key}")
        optimize_options[key] = func(
            model=model,
            func_name=func_name,
            mapping=mapping,
            optimizer=optimizer,
            **optimize_options,
        )

    return optimize_options


def _standardize_simulation_cost_options_jobs_cmpt(jobs_cmpt: str | ListLike, **kwargs) -> np.ndarray:
    if isinstance(jobs_cmpt, (str, list, tuple, np.ndarray)):
        jobs_cmpt = np.array(jobs_cmpt, ndmin=1)

        for i, joc in enumerate(jobs_cmpt):
            if joc.lower().capitalize() in SIGNS:
                jobs_cmpt[i] = joc.lower().capitalize()
            elif joc.lower() in METRICS:
                jobs_cmpt[i] = joc.lower()
            else:
                raise ValueError(
                    f"Unknown component '{joc}' at index {i} in jobs_cmpt cost_options. Choices: {JOBS_CMPT}"
                )

    else:
        raise TypeError("jobs_cmpt cost_options must be a str or ListLike type (List, Tuple, np.ndarray)")

    return jobs_cmpt


def _standardize_simulation_cost_options_wjobs_cmpt(
    jobs_cmpt: np.ndarray, wjobs_cmpt: AlphaNumeric | ListLike, **kwargs
) -> np.ndarray | str:
    if isinstance(wjobs_cmpt, str):
        if wjobs_cmpt == "mean":
            wjobs_cmpt = np.ones(shape=jobs_cmpt.size, dtype=np.float32) / jobs_cmpt.size

        elif wjobs_cmpt == "lquartile":
            wjobs_cmpt = np.ones(shape=jobs_cmpt.size, dtype=np.float32) * np.float32(-0.25)

        elif wjobs_cmpt == "median":
            wjobs_cmpt = np.ones(shape=jobs_cmpt.size, dtype=np.float32) * np.float32(-0.5)

        elif wjobs_cmpt == "uquartile":
            wjobs_cmpt = np.ones(shape=jobs_cmpt.size, dtype=np.float32) * np.float32(-0.75)

        else:
            raise ValueError(
                f"Unknown alias '{wjobs_cmpt}' for wjobs_cmpt cost_options. Choices: {WEIGHT_ALIAS}"
            )

    elif isinstance(wjobs_cmpt, (int, float, list, tuple, np.ndarray)):
        wjobs_cmpt = np.array(wjobs_cmpt, ndmin=1, dtype=np.float32)

        if wjobs_cmpt.size != jobs_cmpt.size:
            raise ValueError(
                f"Inconsistent sizes between jobs_cmpt ({jobs_cmpt.size}) and wjobs_cmpt ({wjobs_cmpt.size}) "
                f"cost_options"
            )

        for i, wjoc in enumerate(wjobs_cmpt):
            if wjoc < 0:
                raise ValueError(f"Negative component {wjoc:f} at index {i} in wjobs_cmpt cost_options")

    else:
        raise TypeError(
            "wjobs_cmpt cost_options must be of AlphaNumeric type (str, int, float) or ListLike type "
            "(List, Tuple, np.ndarray)"
        )

    return wjobs_cmpt


def _standardize_simulation_cost_options_jobs_cmpt_tfm(
    jobs_cmpt: np.ndarray, jobs_cmpt_tfm: str | ListLike, **kwargs
) -> np.ndarray:
    if isinstance(jobs_cmpt_tfm, str):
        if jobs_cmpt_tfm.lower() in JOBS_CMPT_TFM:
            # % Broadcast transformation for all jobs cmpt
            jobs_cmpt_tfm = np.array([jobs_cmpt_tfm.lower()] * jobs_cmpt.size, ndmin=1)
        else:
            raise ValueError(
                f"Unknown discharge transformation '{jobs_cmpt_tfm}' in job_cmpt_tfm cost_options. "
                f"Choices {JOBS_CMPT_TFM}"
            )
    elif isinstance(jobs_cmpt_tfm, (list, tuple, np.ndarray)):
        jobs_cmpt_tfm = np.array(jobs_cmpt_tfm, ndmin=1)

        if jobs_cmpt.size != jobs_cmpt_tfm.size:
            raise ValueError(
                f"Inconsistent sizes between jobs_cmpt ({jobs_cmpt.size}) and jobs_cmpt_tfm "
                f"({jobs_cmpt_tfm.size}) cost_options"
            )

        for i, joct in enumerate(jobs_cmpt_tfm):
            if joct.lower() in JOBS_CMPT_TFM:
                jobs_cmpt_tfm[i] = joct.lower()
            else:
                raise ValueError(
                    f"Unknown discharge transformation '{jobs_cmpt_tfm}' at index {i} in job_cmpt_tfm "
                    f"cost_options. Choices: {JOBS_CMPT_TFM}"
                )
    else:
        raise TypeError("jobs_cmpt_tfm cost_options must be a str or ListLike type (List, Tuple, np.ndarray)")

    # % No transformation applied to signatures
    avail_tfm_signs = ["keep"]

    for i in range(jobs_cmpt_tfm.size):
        joc = jobs_cmpt[i]
        joct = jobs_cmpt_tfm[i]
        if joc in SIGNS and joct not in avail_tfm_signs:
            raise ValueError(
                f"Discharge transformation '{joct}' is not available for component '{joc}' at index {i} in "
                f"jobs_cmpt_tfm cost_options. Choices: {avail_tfm_signs}"
            )

    return jobs_cmpt_tfm


def _standardize_simulation_cost_options_wjreg(wjreg: AlphaNumeric, **kwargs) -> str | float:
    if isinstance(wjreg, str):
        if wjreg.lower() in WJREG_ALIAS:
            wjreg = wjreg.lower()
        else:
            raise ValueError(f"Unknown alias '{wjreg}' for wjreg in cost_options. Choices: {WJREG_ALIAS}")

    elif isinstance(wjreg, (int, float)):
        wjreg = float(wjreg)
        if wjreg < 0:
            raise ValueError("wjreg cost_options must be greater than or equal to 0")
    else:
        raise TypeError("wjreg cost_options must be of AlphaNumeric type (str, int, float)")

    return wjreg


def _standardize_simulation_cost_options_jreg_cmpt(jreg_cmpt: str | ListLike, **kwargs) -> np.ndarray:
    if isinstance(jreg_cmpt, (str, list, tuple, np.ndarray)):
        jreg_cmpt = np.array(jreg_cmpt, ndmin=1)

        for i, jrc in enumerate(jreg_cmpt):
            if jrc.lower() in JREG_CMPT:
                jreg_cmpt[i] = jrc.lower()
            else:
                raise ValueError(
                    f"Unknown component '{jrc}' at index {i} in jreg_cmpt cost_options. Choices: {JREG_CMPT}"
                )
    else:
        raise TypeError("jreg_cmpt cost_options must be a str or ListLike type (List, Tuple, np.ndarray)")

    return jreg_cmpt


def _standardize_simulation_cost_options_wjreg_cmpt(
    jreg_cmpt: np.ndarray, wjreg_cmpt: AlphaNumeric | ListLike, **kwargs
) -> np.ndarray | str:
    if isinstance(wjreg_cmpt, str):
        if wjreg_cmpt == "mean":
            wjreg_cmpt = np.ones(shape=jreg_cmpt.size, dtype=np.float32) / jreg_cmpt.size

        elif wjreg_cmpt == "lquartile":
            wjreg_cmpt = np.ones(shape=jreg_cmpt.size, dtype=np.float32) * np.float32(-0.25)

        elif wjreg_cmpt == "median":
            wjreg_cmpt = np.ones(shape=jreg_cmpt.size, dtype=np.float32) * np.float32(-0.5)

        elif wjreg_cmpt == "uquartile":
            wjreg_cmpt = np.ones(shape=jreg_cmpt.size, dtype=np.float32) * np.float32(-0.75)

        else:
            raise ValueError(
                f"Unknown alias '{wjreg_cmpt}' for wjreg_cmpt cost_options. Choices: {WEIGHT_ALIAS}"
            )

    elif isinstance(wjreg_cmpt, (int, float, list, tuple, np.ndarray)):
        wjreg_cmpt = np.array(wjreg_cmpt, ndmin=1, dtype=np.float32)

        if wjreg_cmpt.size != jreg_cmpt.size:
            raise ValueError(
                f"Inconsistent sizes between jreg_cmpt ({jreg_cmpt.size}) and wjreg_cmpt ({wjreg_cmpt.size}) "
                f"cost_options"
            )

        for i, wjrc in enumerate(wjreg_cmpt):
            if wjrc < 0:
                raise ValueError(f"Negative component {wjrc:f} at index {i} in wjreg_cmpt cost_options")

    else:
        raise TypeError(
            "wjreg_cmpt cost_options must be of AlphaNumeric type (str, int, float) or ListLike type (List, "
            "Tuple, np.ndarray)"
        )

    return wjreg_cmpt


def _standardize_simulation_cost_options_end_warmup(
    model: Model, end_warmup: str | pd.Timestamp | None, **kwargs
) -> pd.Timestamp:
    st = pd.Timestamp(model.setup.start_time)
    et = pd.Timestamp(model.setup.end_time)

    if end_warmup is None:
        end_warmup = pd.Timestamp(st)

    else:
        if isinstance(end_warmup, str):
            try:
                end_warmup = pd.Timestamp(end_warmup)

            except Exception:
                raise ValueError(f"end_warmup '{end_warmup}' cost_options is an invalid date") from None

        elif isinstance(end_warmup, pd.Timestamp):
            pass

        else:
            raise TypeError("end_warmup cost_options must be str or pandas.Timestamp object")

        if (end_warmup - st).total_seconds() < 0 or (et - end_warmup).total_seconds() < 0:
            raise ValueError(
                f"end_warmup '{end_warmup}' cost_options must be between start time '{st}' and end time "
                f"'{et}'"
            )

    return end_warmup


def _standardize_simulation_cost_options_gauge(
    model: Model,
    func_name: str,
    gauge: str | ListLike,
    end_warmup: pd.Timestamp,
    **kwargs,
) -> np.ndarray:
    if isinstance(gauge, str):
        if gauge == "dws":
            gauge = np.empty(shape=0)

            for i, pos in enumerate(model.mesh.gauge_pos):
                if model.mesh.flwdst[tuple(pos)] == 0:
                    gauge = np.append(gauge, model.mesh.code[i])

        elif gauge == "all":
            gauge = np.array(model.mesh.code, ndmin=1)

        elif gauge in model.mesh.code:
            gauge = np.array(gauge, ndmin=1)

        else:
            raise ValueError(
                f"Unknown alias or gauge code '{gauge}' for gauge cost_options. "
                f"Choices: {GAUGE_ALIAS + list(model.mesh.code)}"
            )

    elif isinstance(gauge, (list, tuple, np.ndarray)):
        gauge_bak = np.array(gauge, ndmin=1)
        gauge = np.empty(shape=0)

        for ggc in gauge_bak:
            if ggc in model.mesh.code:
                gauge = np.append(gauge, ggc)
            else:
                raise ValueError(
                    f"Unknown gauge code '{ggc}' in gauge cost_options. Choices: {list(model.mesh.code)}"
                )

    else:
        raise TypeError("gauge cost_options must be a str or ListLike type (List, Tuple, np.ndarray)")

    # % Check that there is observed discharge available when optimizing. No particular issue with a
    # % forward run
    if "optimize" in func_name:
        st = pd.Timestamp(model.setup.start_time)
        et = pd.Timestamp(model.setup.end_time)
        start_slice = int((end_warmup - st).total_seconds() / model.setup.dt)
        end_slice = model.setup.ntime_step
        time_slice = slice(start_slice, end_slice)
        for i, ggc in enumerate(model.mesh.code):
            if ggc in gauge:
                if np.all(model.response_data.q[i, time_slice] < 0):
                    raise ValueError(
                        f"No observed discharge available at gauge '{ggc}' for the selected "
                        f"optimization period ['{end_warmup}', '{et}']"
                    )

    return gauge


def _standardize_simulation_cost_options_wgauge(
    gauge: np.ndarray, wgauge: AlphaNumeric | ListLike, **kwargs
) -> np.ndarray | str:
    if isinstance(wgauge, str):
        if wgauge == "mean":
            wgauge = np.ones(shape=gauge.size, dtype=np.float32)
            if wgauge.size > 0:
                wgauge *= 1 / wgauge.size

        elif wgauge == "lquartile":
            wgauge = np.ones(shape=gauge.size, dtype=np.float32) * np.float32(-0.25)

        elif wgauge == "median":
            wgauge = np.ones(shape=gauge.size, dtype=np.float32) * np.float32(-0.5)

        elif wgauge == "uquartile":
            wgauge = np.ones(shape=gauge.size, dtype=np.float32) * np.float32(-0.75)

        else:
            raise ValueError(f"Unknown alias '{wgauge}' for wgauge cost_options. Choices: {WEIGHT_ALIAS}")

    elif isinstance(wgauge, (int, float, list, tuple, np.ndarray)):
        wgauge = np.array(wgauge, ndmin=1)

        if wgauge.size != gauge.size:
            raise ValueError(
                f"Inconsistent sizes between wgauge ({wgauge.size}) and gauge ({gauge.size}) cost_options"
            )

        for wggc in wgauge:
            if wggc < 0:
                raise ValueError(f"Negative component {wggc:f} in wgauge cost_options")

    else:
        raise TypeError(
            "wgauge cost_options must be of AlphaNumeric type (str, int, float) or ListLike type (List, "
            "Tuple, np.ndarray)"
        )

    return wgauge


def _standardize_simulation_cost_options_event_seg(event_seg: dict, **kwargs) -> dict:
    if isinstance(event_seg, dict):
        simulation_event_seg_keys = EVENT_SEG_KEYS.copy()
        simulation_event_seg_keys.remove("by")

        for key, value in event_seg.items():
            if key in simulation_event_seg_keys:
                func = eval(f"_standardize_hydrograph_segmentation_{key}")
                event_seg[key] = func(value)

            else:
                raise ValueError(
                    f"Unknown event_seg key '{key}' in cost_options. Choices: {simulation_event_seg_keys}"
                )

    else:
        raise TypeError("event_seg cost_options must be a dictionary or None")

    return event_seg


# % Some standardization of control_prior must be done with Fortran calls
# % Otherwise we have to compute control size in Python ...
def _standardize_simulation_cost_options_control_prior(control_prior: dict | None, **kwargs) -> dict | None:
    return control_prior


def _standardize_simulation_cost_options(model: Model, func_name: str, cost_options: dict | None) -> dict:
    if cost_options is None:
        cost_options = DEFAULT_SIMULATION_COST_OPTIONS[func_name].copy()

    else:
        if isinstance(cost_options, dict):
            pop_keys = []
            for key, _ in cost_options.items():
                if key not in DEFAULT_SIMULATION_COST_OPTIONS[func_name]:
                    pop_keys.append(key)
                    warnings.warn(
                        f"Unknown cost_options key '{key}'. "
                        f"Choices: {list(DEFAULT_SIMULATION_COST_OPTIONS[func_name].keys())}",
                        stacklevel=2,
                    )

            for key in pop_keys:
                cost_options.pop(key)

        else:
            raise TypeError("cost_options argument must be a dictionary")

    for key, value in DEFAULT_SIMULATION_COST_OPTIONS[func_name].items():
        cost_options.setdefault(key, value)
        func = eval(f"_standardize_simulation_cost_options_{key}")
        cost_options[key] = func(model=model, func_name=func_name, **cost_options)

    return cost_options


def _standardize_simulation_common_options_ncpu(ncpu: Numeric) -> int:
    if isinstance(ncpu, (int, float)):
        ncpu = int(ncpu)
        if ncpu <= 0:
            raise ValueError("ncpu common_options must be greater than 0")
        elif ncpu > os.cpu_count():
            ncpu = os.cpu_count()
            warnings.warn(
                f"ncpu common_options cannot be greater than the number of CPU(s) on the machine. ncpu is "
                f"set to {os.cpu_count()}",
                stacklevel=2,
            )
    else:
        raise TypeError("ncpu common_options must be of Numeric type (int, float)")

    # Parallel computation not supported on Windows
    if platform.system() == "Windows" and ncpu > 1:
        ncpu = 1
        warnings.warn("Parallel computation is not supported on Windows. ncpu is set to 1", stacklevel=2)

    return ncpu


def _standardize_simulation_common_options_verbose(verbose: bool) -> bool:
    if isinstance(verbose, bool):
        pass
    else:
        raise TypeError("verbose common_options must be boolean")

    return verbose


def _standardize_simulation_common_options(common_options: dict | None) -> dict:
    if common_options is None:
        common_options = DEFAULT_SIMULATION_COMMON_OPTIONS.copy()
    else:
        if isinstance(common_options, dict):
            pop_keys = []
            for key in common_options:
                if key not in DEFAULT_SIMULATION_COMMON_OPTIONS:
                    pop_keys.append(key)
                    warnings.warn(
                        f"Unknown common_options key '{key}': "
                        f"Choices: {list(DEFAULT_SIMULATION_COMMON_OPTIONS.keys())}",
                        stacklevel=2,
                    )

            for key in pop_keys:
                common_options.pop(key)

        else:
            raise TypeError("common_options argument must be a dictionary")

    for key, value in DEFAULT_SIMULATION_COMMON_OPTIONS.items():
        common_options.setdefault(key, value)
        func = eval(f"_standardize_simulation_common_options_{key}")
        common_options[key] = func(common_options[key])

    return common_options


def _standardize_simulation_return_options_bool(key: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{key} return_options must be boolean")

    return value


def _standardize_simulation_return_options_time_step(
    model: Model, time_step: str | pd.Timestamp | pd.DatetimeIndex | ListLike
) -> pd.DatetimeIndex:
    st = pd.Timestamp(model.setup.start_time)
    et = pd.Timestamp(model.setup.end_time)

    if isinstance(time_step, str):
        if time_step == "all":
            time_step = pd.date_range(start=st, end=et, freq=f"{int(model.setup.dt)}s")[1:]
        else:
            try:
                # % Pass to list to convert to pd.DatetimeIndex
                time_step = [pd.Timestamp(time_step)]

            except Exception:
                raise ValueError(f"time_step '{time_step}' return_options is an invalid date") from None

    elif isinstance(time_step, pd.Timestamp):
        # % Pass to list to convert to pd.DatetimeIndex
        time_step = [time_step]

    elif isinstance(time_step, (list, tuple, np.ndarray)):
        time_step = list(time_step)
        for i, date in enumerate(time_step):
            try:
                time_step[i] = pd.Timestamp(str(date))
            except Exception:
                raise ValueError(f"Invalid date '{date}' at index {i} in time_step return_options") from None
    elif isinstance(time_step, pd.DatetimeIndex):
        pass

    else:
        raise TypeError(
            "time_step return_options must be a str, pd.Timestamp, pd.DatetimeIndex or ListLike type (List, "
            "Tuple, np.ndarray)"
        )

    time_step = pd.DatetimeIndex(time_step)

    # % Check that all dates are inside Model start_time and end_time
    for i, date in enumerate(time_step):
        if (date - st).total_seconds() <= 0 or (et - date).total_seconds() < 0:
            raise ValueError(
                f"date '{date}' at index {i} in time_step return_options must be between start time '{st}' "
                f"and end time '{et}'"
            )

    return time_step


def _standardize_simulation_return_options(model: Model, func_name: str, return_options: dict | None) -> dict:
    if return_options is None:
        return_options = DEFAULT_SIMULATION_RETURN_OPTIONS[func_name].copy()
    else:
        if isinstance(return_options, dict):
            pop_keys = []
            for key in return_options:
                if key not in DEFAULT_SIMULATION_RETURN_OPTIONS[func_name]:
                    pop_keys.append(key)
                    warnings.warn(
                        f"Unknown return_options key '{key}': "
                        f"Choices: {list(DEFAULT_SIMULATION_RETURN_OPTIONS[func_name].keys())}",
                        stacklevel=2,
                    )

            for key in pop_keys:
                return_options.pop(key)

        else:
            raise TypeError("return_options argument must be a dictionary")

    for key, value in DEFAULT_SIMULATION_RETURN_OPTIONS[func_name].items():
        return_options.setdefault(key, value)
        if key == "time_step":
            return_options[key] = _standardize_simulation_return_options_time_step(model, return_options[key])
        else:
            _standardize_simulation_return_options_bool(key, return_options[key])

    return return_options


def _standardize_simulation_parameters_feasibility(model: Model):
    mask = model.mesh.active_cell == 1
    for key in model.rr_parameters.keys:
        arr = model.get_rr_parameters(key)
        low, upp = FEASIBLE_RR_PARAMETERS[key]
        # Do not check if a value is inside the feasible domain outside of active cells
        low_arr = np.min(arr, where=mask, initial=np.inf)
        upp_arr = np.max(arr, where=mask, initial=-np.inf)

        if (low_arr + F_PRECISION) <= low or (upp_arr - F_PRECISION) >= upp:
            raise ValueError(
                f"Invalid value for model rr_parameter '{key}'. rr_parameter domain [{low_arr}, {upp_arr}] "
                f"is not included in the feasible domain ]{low}, {upp}["
            )

    for key in model.rr_initial_states.keys:
        arr = model.get_rr_initial_states(key)
        low, upp = FEASIBLE_RR_INITIAL_STATES[key]
        # Do not check if a value is inside the feasible domain outside of active cells
        low_arr = np.min(arr, where=mask, initial=np.inf)
        upp_arr = np.max(arr, where=mask, initial=-np.inf)

        if (low_arr + F_PRECISION) <= low or (upp_arr - F_PRECISION) >= upp:
            raise ValueError(
                f"Invalid value for model rr_initial_state '{key}'. rr_initial_state domain "
                f"[{low_arr}, {upp_arr}] is not included in the feasible domain ]{low}, {upp}["
            )

    for key in model.serr_mu_parameters.keys:
        arr = model.get_serr_mu_parameters(key)
        # % Skip if size == 0, i.e. no gauge
        if arr.size == 0:
            continue
        low, upp = FEASIBLE_SERR_MU_PARAMETERS[key]
        low_arr = np.min(arr)
        upp_arr = np.max(arr)

        if (low_arr + F_PRECISION) <= low or (upp_arr - F_PRECISION) >= upp:
            raise ValueError(
                f"Invalid value for model serr_mu_parameter '{key}'. serr_mu_parameter domain "
                f"[{low_arr}, {upp_arr}] is not included in the feasible domain ]{low}, {upp}["
            )

    for key in model.serr_sigma_parameters.keys:
        arr = model.get_serr_sigma_parameters(key)
        # % Skip if size == 0, i.e. no gauge
        if arr.size == 0:
            continue
        low, upp = FEASIBLE_SERR_SIGMA_PARAMETERS[key]
        low_arr = np.min(arr)
        upp_arr = np.max(arr)

        if (low_arr + F_PRECISION) <= low or (upp_arr - F_PRECISION) >= upp:
            raise ValueError(
                f"Invalid value for model serr_sigma_parameter '{key}'. serr_sigma_parameter domain "
                f"[{low_arr}, {upp_arr}] is not included in the feasible domain ]{low}, {upp}["
            )


def _standardize_simulation_optimize_options_finalize(
    model: Model, mapping: str, optimizer: str, optimize_options: dict
) -> dict:
    optimize_options["mapping"] = mapping
    optimize_options["optimizer"] = optimizer

    # % Check if decriptors are not found for regionalization mappings
    if model.setup.nd == 0 and mapping in REGIONAL_MAPPING:
        raise ValueError(
            f"Physiographic descriptors are required for optimization with {mapping} mapping. "
            f"Please check if read_descriptor, descriptor_name and descriptor_directory "
            f"are properly defined in the model setup."
        )

    descriptor_present = "descriptor" in optimize_options

    # % Handle parameters
    # % rr parameters
    optimize_options["rr_parameters"] = np.zeros(shape=model.setup.nrrp, dtype=np.int32)
    optimize_options["l_rr_parameters"] = np.zeros(shape=model.setup.nrrp, dtype=np.float32)
    optimize_options["u_rr_parameters"] = np.zeros(shape=model.setup.nrrp, dtype=np.float32)

    if descriptor_present:
        optimize_options["rr_parameters_descriptor"] = np.zeros(
            shape=(model.setup.nd, model.setup.nrrp), dtype=np.int32
        )

    for i, key in enumerate(model.rr_parameters.keys):
        if key in optimize_options["parameters"]:
            optimize_options["rr_parameters"][i] = 1
            optimize_options["l_rr_parameters"][i] = optimize_options["bounds"][key][0]
            optimize_options["u_rr_parameters"][i] = optimize_options["bounds"][key][1]

            if descriptor_present:
                for j, desc in enumerate(model.setup.descriptor_name):
                    if desc in optimize_options["descriptor"][key]:
                        optimize_options["rr_parameters_descriptor"][j, i] = 1

    # % rr initial states
    optimize_options["rr_initial_states"] = np.zeros(shape=model.setup.nrrs, dtype=np.int32)
    optimize_options["l_rr_initial_states"] = np.zeros(shape=model.setup.nrrs, dtype=np.float32)
    optimize_options["u_rr_initial_states"] = np.zeros(shape=model.setup.nrrs, dtype=np.float32)
    if descriptor_present:
        optimize_options["rr_initial_states_descriptor"] = np.zeros(
            shape=(model.setup.nd, model.setup.nrrs), dtype=np.int32
        )

    for i, key in enumerate(model.rr_initial_states.keys):
        if key in optimize_options["parameters"]:
            optimize_options["rr_initial_states"][i] = 1
            optimize_options["l_rr_initial_states"][i] = optimize_options["bounds"][key][0]
            optimize_options["u_rr_initial_states"][i] = optimize_options["bounds"][key][1]

            if descriptor_present:
                for j, desc in enumerate(model.setup.descriptor_name):
                    if desc in optimize_options["descriptor"][key]:
                        optimize_options["rr_initial_states_descriptor"][j, i] = 1

    # % nn parameters
    optimize_options["nn_parameters"] = np.zeros(shape=len(NN_PARAMETERS_KEYS), dtype=np.int32)

    for i, key in enumerate(NN_PARAMETERS_KEYS):
        if key in optimize_options["parameters"]:
            optimize_options["nn_parameters"][i] = 1

    # % serr mu parameters
    optimize_options["serr_mu_parameters"] = np.zeros(shape=model.setup.nsep_mu, dtype=np.int32)
    optimize_options["l_serr_mu_parameters"] = np.zeros(shape=model.setup.nsep_mu, dtype=np.float32)
    optimize_options["u_serr_mu_parameters"] = np.zeros(shape=model.setup.nsep_mu, dtype=np.float32)

    for i, key in enumerate(model.serr_mu_parameters.keys):
        if key in optimize_options["parameters"]:
            optimize_options["serr_mu_parameters"][i] = 1
            optimize_options["l_serr_mu_parameters"][i] = optimize_options["bounds"][key][0]
            optimize_options["u_serr_mu_parameters"][i] = optimize_options["bounds"][key][1]

    # % serr sigma parameters
    optimize_options["serr_sigma_parameters"] = np.zeros(shape=model.setup.nsep_sigma, dtype=np.int32)
    optimize_options["l_serr_sigma_parameters"] = np.zeros(shape=model.setup.nsep_sigma, dtype=np.float32)
    optimize_options["u_serr_sigma_parameters"] = np.zeros(shape=model.setup.nsep_sigma, dtype=np.float32)

    for i, key in enumerate(model.serr_sigma_parameters.keys):
        if key in optimize_options["parameters"]:
            optimize_options["serr_sigma_parameters"][i] = 1
            optimize_options["l_serr_sigma_parameters"][i] = optimize_options["bounds"][key][0]
            optimize_options["u_serr_sigma_parameters"][i] = optimize_options["bounds"][key][1]

    return optimize_options


def _standardize_simulation_cost_options_finalize(model: Model, func_name: str, cost_options: dict) -> dict:
    is_bayesian = "bayesian" in func_name

    if is_bayesian:
        cost_options["bayesian"] = True

    cost_options["njoc"] = cost_options["jobs_cmpt"].size if "jobs_cmpt" in cost_options else 0
    cost_options["njrc"] = cost_options["jreg_cmpt"].size if "jreg_cmpt" in cost_options else 0

    if any(f.startswith("E") for f in cost_options.get("jobs_cmpt", [])):
        info_event = _mask_event(model=model, **cost_options["event_seg"])
        cost_options["n_event"] = info_event["n"]
        cost_options["mask_event"] = info_event["mask"]

    if isinstance(cost_options.get("wjreg", None), str):
        cost_options["auto_wjreg"] = cost_options["wjreg"]
        cost_options["wjreg"] = 0

    # % Handle flags send to Fortran
    # % gauge and wgauge
    gauge = np.zeros(shape=model.mesh.ng, dtype=np.int32)
    if not is_bayesian:
        wgauge = np.zeros(shape=model.mesh.ng, dtype=np.float32)

    n = 0
    for i, gc in enumerate(model.mesh.code):
        if gc in cost_options["gauge"]:
            gauge[i] = 1
            if not is_bayesian:
                wgauge[i] = cost_options["wgauge"][n]
            n += 1

    cost_options["nog"] = np.count_nonzero(gauge)
    cost_options["gauge"] = gauge
    if not is_bayesian:
        cost_options["wgauge"] = wgauge

    # % end_warmup
    st = pd.Timestamp(model.setup.start_time)
    cost_options["end_warmup"] = int((cost_options["end_warmup"] - st).total_seconds() / model.setup.dt)

    return cost_options


def _standardize_simulation_return_options_finalize(model: Model, return_options: dict):
    st = pd.Timestamp(model.setup.start_time)

    mask_time_step = np.zeros(shape=model.setup.ntime_step, dtype=bool)
    time_step_to_returns_time_step = np.zeros(shape=model.setup.ntime_step, dtype=np.int32) - np.int32(99)

    for date in return_options["time_step"]:
        ind = int((date - st).total_seconds() / model.setup.dt) - 1
        mask_time_step[ind] = True

    ind = 0
    for i in range(model.setup.ntime_step):
        if mask_time_step[i]:
            time_step_to_returns_time_step[i] = ind
            ind += 1

    # % To pass character array to Fortran.
    keys = [k for k, v in return_options.items() if k != "time_step" and v]
    if keys:
        max_len = max(map(len, keys))
    else:
        max_len = 0
    fkeys = np.empty(shape=(max_len, len(keys)), dtype="c")

    for i, key in enumerate(keys):
        fkeys[:, i] = key + (max_len - len(key)) * " "

    return_options.update(
        {
            "nmts": np.count_nonzero(mask_time_step),
            "mask_time_step": mask_time_step,
            "time_step_to_returns_time_step": time_step_to_returns_time_step,
            "fkeys": fkeys,
            "keys": keys,
        }
    )

    pop_keys = [
        k
        for k in return_options
        if k not in ["nmts", "mask_time_step", "time_step_to_returns_time_step", "time_step", "fkeys", "keys"]
    ]
    for key in pop_keys:
        return_options.pop(key)


def _standardize_default_optimize_options_args(mapping: str, optimizer: str | None) -> AnyTuple:
    mapping = _standardize_simulation_mapping(mapping)

    optimizer = _standardize_simulation_optimizer(mapping, optimizer)

    return (mapping, optimizer)


def _standardize_default_bayesian_optimize_options_args(mapping: str, optimizer: str | None) -> AnyTuple:
    return _standardize_default_optimize_options_args(mapping, optimizer)
