from __future__ import annotations

from smash._constant import (
    STRUCTURE_OPR_PARAMETERS,
    STRUCTURE_OPR_STATES,
    OPTIMIZABLE_OPR_PARAMETERS,
    OPTIMIZABLE_OPR_INITIAL_STATES,
    FEASIBLE_OPR_PARAMETERS,
    FEASIBLE_OPR_INITIAL_STATES,
    MAPPING,
    MAPPING_OPTIMIZER,
    PY_OPTIMIZER_CLASS,
    PY_OPTIMIZER,
    F90_OPTIMIZER_CONTROL_TFM,
    SIMULATION_OPTIMIZE_OPTIONS_KEYS,
    DEFAULT_TERMINATION_CRIT,
    DEFAULT_SIMULATION_COMMON_OPTIONS,
    DEFAULT_SIMULATION_COST_OPTIONS,
    DEFAULT_SIMULATION_RETURN_OPTIONS,
    METRICS,
    SIGNS,
    JOBS_CMPT,
    JREG_CMPT,
    WEIGHT_ALIAS,
    GAUGE_ALIAS,
    EVENT_SEG_KEYS,
)

from smash.core.signal_analysis.segmentation._standardize import (
    _standardize_hydrograph_segmentation_peak_quant,
    _standardize_hydrograph_segmentation_max_duration,
)

from smash.core.signal_analysis.segmentation._tools import _mask_event

from smash.factory.net.net import Net

from smash.factory.net._optimizers import Adam, Adagrad, SGD, RMSprop

from smash.factory.samples.samples import Samples

import numpy as np
import pandas as pd
import warnings
import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash._typing import Numeric, ListLike
    from smash.core.model.model import Model


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
                    f"Unknown optimizer '{optimizer}' for mapping '{mapping}'. Choices: {MAPPING_OPTIMIZER[mapping]}"
                )

        else:
            raise TypeError("optimizer argument must be a str")

    return optimizer.lower()


def _standardize_simulation_samples(model: Model, samples: Samples) -> Samples:
    if isinstance(samples, Samples):
        for key in samples._problem["names"]:
            if key in model.opr_parameters.keys:
                l, u = FEASIBLE_OPR_PARAMETERS[key]
            elif key in model.opr_initial_states.keys:
                l, u = FEASIBLE_OPR_INITIAL_STATES[key]
            else:
                available_parameters = list(model.opr_parameters.keys) + list(
                    model.opr_initial_states.keys
                )
                raise ValueError(
                    f"Unknown parameter '{key}' in samples attributes. Choices: {available_parameters}"
                )

            # % Check that sample is inside feasible domain
            arr = getattr(samples, key)
            l_arr = np.min(arr)
            u_arr = np.max(arr)
            if l_arr <= l or u_arr >= u:
                raise ValueError(
                    f"Invalid sample values for parameter '{key}'. Sample domain [{l_arr}, {u_arr}] is not included in the feasible domain ]{l}, {u}["
                )
    else:
        raise TypeError(f"samples arguments must be a smash.Samples object")

    return samples


def _standardize_simulation_optimize_options_parameters(
    model: Model, parameters: str | ListLike | None, **kwargs
) -> np.ndarray:
    available_opr_parameters = [
        key
        for key in STRUCTURE_OPR_PARAMETERS[model.setup.structure]
        if OPTIMIZABLE_OPR_PARAMETERS[key]
    ]
    available_opr_initial_states = [
        key
        for key in STRUCTURE_OPR_STATES[model.setup.structure]
        if OPTIMIZABLE_OPR_INITIAL_STATES[key]
    ]
    available_parameters = available_opr_parameters + available_opr_initial_states

    if parameters is None:
        parameters = np.array(available_opr_parameters, ndmin=1)

    else:
        if isinstance(parameters, (str, list, tuple, np.ndarray)):
            parameters = np.array(parameters, ndmin=1)

            for i, prmt in enumerate(parameters):
                if prmt in available_parameters:
                    parameters[i] = prmt.lower()

                else:
                    raise ValueError(
                        f"Unknown or non optimizable parameter '{prmt}' at index {i} in parameters optimize_options. Choices: {available_parameters}"
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
    if bounds is None:
        bounds = {}

    else:
        if isinstance(bounds, dict):
            for key, value in bounds.items():
                if key in parameters:
                    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2:
                        if value[0] >= value[1]:
                            raise ValueError(
                                f"Lower bound value {value[0]} is greater than or equal to upper bound {value[1]} for parameter '{key}' in bounds optimize_options"
                            )
                        else:
                            bounds[key] = tuple(value)
                    else:
                        raise TypeError(
                            f"Bounds values for parameter '{key}' must be of ListLike type (List, Tuple, np.ndarray) of size 2 in bounds optimize_options"
                        )
                else:
                    raise ValueError(
                        f"Unknown or non optimized parameter '{key}' in bounds optimize_options. Choices: {parameters}"
                    )
        else:
            TypeError("bounds optimize_options must be a dictionary")

    parameters_bounds = dict(
        **model.get_opr_parameters_bounds(), **model.get_opr_initial_states_bounds()
    )
    for key, value in parameters_bounds.items():
        if key in parameters:
            bounds.setdefault(key, value)

    # % Check that bounds are inside feasible domain and that bounds include parameter domain
    for key, value in bounds.items():
        if key in model.opr_parameters.keys:
            arr = model.get_opr_parameters(key)
            l, u = FEASIBLE_OPR_PARAMETERS[key]
        elif key in model.opr_initial_states.keys:
            arr = model.get_opr_initial_states(key)
            l, u = FEASIBLE_OPR_INITIAL_STATES[key]
        # % In case we have other kind of parameters. Should be unreachable.
        else:
            pass

        l_arr = np.min(arr)
        u_arr = np.max(arr)
        if (l_arr + 1e-3) < value[0] or (u_arr - 1e-3) > value[1]:
            raise ValueError(
                f"Invalid bounds values for parameter '{key}'. Bounds domain [{value[0]}, {value[1]}] does not include parameter domain [{l_arr}, {u_arr}] in bounds optimize_options"
            )

        if value[0] <= l or value[1] >= u:
            raise ValueError(
                f"Invalid bounds values for parameter '{key}'. Bounds domain [{value[0]}, {value[1]}] is not included in the feasible domain ]{l}, {u}[ in bounds optimize_options"
            )

    return bounds


def _standardize_simulation_optimize_options_control_tfm(
    optimizer: str, control_tfm: str | None, **kwargs
) -> (str, None):
    if control_tfm is None:
        control_tfm = F90_OPTIMIZER_CONTROL_TFM[optimizer][0]
    else:
        if isinstance(control_tfm, str):
            if control_tfm.lower() not in F90_OPTIMIZER_CONTROL_TFM[optimizer]:
                raise ValueError(
                    f"Unknown transformation '{control_tfm}' in control_tfm optimize_options. Choices: {F90_OPTIMIZER_CONTROL_TFM[optimizer]}"
                )
        else:
            TypeError("control_tfm optimize_options must be a str")

    return control_tfm


def _standardize_simulation_optimize_options_descriptor(
    model: Model, parameters: np.ndarray, descriptor: dict | None, **kwargs
) -> dict:
    if descriptor is None:
        descriptor = {}

    else:
        if isinstance(descriptor, dict):
            for key, value in descriptor.items():
                if key in parameters:
                    if isinstance(value, (str, list, tuple, np.ndarray)):
                        value = np.array(value, ndmin=1)
                        for vle in value:
                            if vle not in model.setup.descriptor_name:
                                raise ValueError(
                                    f"Unknown descriptor '{vle}' for parameter '{key}' in descriptor optimize_options. Choices: {list(model.setup.descriptor_name)}"
                                )
                        descriptor[key] = value
                    else:
                        raise TypeError(
                            f"Descriptor values for parameter '{key}' must be a str or ListLike type (List, Tuple, np.ndarray) in descriptor optimize_options"
                        )
                else:
                    raise ValueError(
                        f"Unknown or non optimized parameter '{key}' in descriptor optimize_options. Choices: {parameters}"
                    )
        else:
            raise TypeError("descriptor optimize_options must be a dictionary")

    for prmt in parameters:
        descriptor.setdefault(prmt, model.setup.descriptor_name)

    # % Check that descriptors are not uniform
    for i, desc in enumerate(model.setup.descriptor_name):
        arr = model.physio_data.descriptor[..., i]
        if np.all(arr == arr[0, 0]):
            prmt_err = []
            for key, value in descriptor.items():
                if desc in value:
                    prmt_err.append(key)
            raise ValueError(
                f"Spatially uniform descriptor '{desc}' found for parameter(s) {prmt_err} in descriptor optimize_options. Must be removed to perform optimization"
            )

    return descriptor


def _standardize_simulation_optimize_options_net(
    model: Model, bounds: dict, net: Net | None, **kwargs
) -> Net:
    bound_values = list(bounds.values())
    ncv = len(bound_values)

    nd = model.setup.nd

    active_mask = np.where(model.mesh.active_cell == 1)
    ntrain = active_mask[0].shape[0]

    if net is None:  # default graph
        net = Net()

        n_neurons = round(np.sqrt(ntrain * nd) * 2 / 3)

        net.add(
            layer="dense",
            options={
                "input_shape": (nd,),
                "neurons": n_neurons,
                "kernel_initializer": "glorot_uniform",
            },
        )
        net.add(layer="activation", options={"name": "relu"})

        net.add(
            layer="dense",
            options={
                "neurons": round(n_neurons / 2),
                "kernel_initializer": "glorot_uniform",
            },
        )
        net.add(layer="activation", options={"name": "relu"})

        net.add(
            layer="dense",
            options={"neurons": ncv, "kernel_initializer": "glorot_uniform"},
        )
        net.add(layer="activation", options={"name": "sigmoid"})

        net.add(
            layer="scale",
            options={"bounds": bound_values},
        )

    elif not isinstance(net, Net):
        raise TypeError(f"net optimize_options: Unknown network {net}")

    elif not net.layers:
        raise ValueError(f"net optimize_options: The graph has not been set yet")

    else:
        # % check input shape
        ips = net.layers[0].input_shape

        if ips[0] != nd:
            raise ValueError(
                f"net optimize_options: Inconsistent value between the number of input layer ({ips[0]}) and the number of descriptors ({nd}): {ips[0]} != {nd}"
            )

        # % check output shape
        ios = net.layers[-1].output_shape()

        if ios[0] != ncv:
            raise ValueError(
                f"net optimize_options: Inconsistent value between the number of output layer ({ios[0]}) and the number of parameters ({ncv}): {ios[0]} != {ncv}"
            )

        # % check bounds constraints
        if hasattr(net.layers[-1], "_scale_func"):
            net_bounds = net.layers[-1]._scale_func._bounds

            diff = np.not_equal(net_bounds, bound_values)

            for i, name in enumerate(bounds.keys()):
                if diff[i].any():
                    warnings.warn(
                        f"net optimize_options: Inconsistent value(s) between the bound in scaling layer and the parameter bound for {name}: {net_bounds[i]} != {bound_values[i]}. Ignoring default bounds for scaling layer."
                    )

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
            opt_class = eval(PY_OPTIMIZER_CLASS[PY_OPTIMIZER.index(optimizer)])
            learning_rate = opt_class().learning_rate

        else:
            raise TypeError(
                "learning_rate optimize_options must be of Numeric type (int, float) or None"
            )

    return learning_rate


def _standardize_simulation_optimize_options_random_state(
    random_state: Numeric | None, **kwargs
) -> int:
    if random_state is None:
        pass

    else:
        if not isinstance(random_state, (int, float)):
            raise TypeError(
                "random_state optimize_options must be of Numeric type (int, float)"
            )

        random_state = int(random_state)

        if random_state < 0 or random_state > 4_294_967_295:
            raise ValueError(
                "random_state optimize_options must be between 0 and 2**32 - 1"
            )

    return random_state


def _standardize_simulation_optimize_options_termination_crit(
    optimizer: str, termination_crit: dict | None, **kwargs
) -> dict:
    if termination_crit is None:
        termination_crit = DEFAULT_TERMINATION_CRIT[optimizer].copy()

    else:
        if isinstance(termination_crit, dict):
            pop_keys = []
            for key, value in termination_crit.items():
                if key not in DEFAULT_TERMINATION_CRIT[optimizer].keys():
                    pop_keys.append(key)
                    warnings.warn(
                        f"Unknown termination_crit key '{key}' for optimizer '{optimizer}'. Choices: {list(DEFAULT_TERMINATION_CRIT[optimizer].keys())}"
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


def _standardize_simulation_optimize_options_termination_crit_maxiter(
    maxiter: Numeric, **kwargs
) -> int:
    if isinstance(maxiter, (int, float)):
        maxiter = int(maxiter)
        if maxiter < 0:
            raise ValueError(
                "maxiter termination_crit must be greater than or equal to 0"
            )
    else:
        raise TypeError("maxiter termination_crit must be of Numeric type (int, float)")

    return maxiter


def _standardize_simulation_optimize_options_termination_crit_factr(
    factr: Numeric, **kwargs
) -> float:
    if isinstance(factr, (int, float)):
        factr = float(factr)
        if factr <= 0:
            raise ValueError("factr termination_crit must be greater than 0")
    else:
        raise TypeError("factr termination_crit must be of Numeric type (int, float)")

    return factr


def _standardize_simulation_optimize_options_termination_crit_pgtol(
    pgtol: Numeric, **kwargs
) -> float:
    if isinstance(pgtol, (int, float)):
        pgtol = float(pgtol)
        if pgtol <= 0:
            raise ValueError("pgtol termination_crit must be greater than 0")
    else:
        raise TypeError("pgtol termination_crit must be of Numeric type (int, float)")

    return pgtol


def _standardize_simulation_optimize_options_termination_crit_epochs(
    epochs: Numeric, **kwargs
) -> float:
    if isinstance(epochs, (int, float)):
        epochs = int(epochs)
        if epochs < 0:
            raise ValueError(
                "epochs termination_crit must be greater than or equal to 0"
            )
    else:
        raise TypeError("epochs termination_crit must be of Numeric type (int, float)")

    return epochs


def _standardize_simulation_optimize_options_termination_crit_early_stopping(
    early_stopping: Numeric, **kwargs
) -> int:
    if isinstance(early_stopping, (int, float)):
        early_stopping = int(early_stopping)

        if early_stopping < 0:
            raise ValueError("early_stopping termination_crit must be non-negative")
    else:
        raise TypeError(
            "early_stopping termination_crit must be of Numeric type (int, float)"
        )

    return early_stopping


def _standardize_simulation_optimize_options(
    model: Model, mapping: str, optimizer: str, optimize_options: dict | None
) -> dict:
    if optimize_options is None:
        optimize_options = dict.fromkeys(
            SIMULATION_OPTIMIZE_OPTIONS_KEYS[(mapping, optimizer)], None
        )

    else:
        if isinstance(optimize_options, dict):
            pop_keys = []
            for key, value in optimize_options.items():
                if key not in SIMULATION_OPTIMIZE_OPTIONS_KEYS[(mapping, optimizer)]:
                    pop_keys.append(key)
                    warnings.warn(
                        f"Unknown optimize_options key '{key}' for mapping '{mapping}' and optimizer '{optimizer}'. Choices: {SIMULATION_OPTIMIZE_OPTIONS_KEYS[(mapping, optimizer)]}"
                    )

            for key in pop_keys:
                optimize_options.pop(key)

        else:
            raise TypeError("optimize_options argument must be a dictionary")

    for key in SIMULATION_OPTIMIZE_OPTIONS_KEYS[(mapping, optimizer)]:
        optimize_options.setdefault(key, None)
        func = eval(f"_standardize_simulation_optimize_options_{key}")
        optimize_options[key] = func(
            model=model, mapping=mapping, optimizer=optimizer, **optimize_options
        )

    return optimize_options


def _standardize_simulation_cost_options_jobs_cmpt(
    jobs_cmpt: str | ListLike, **kwargs
) -> np.ndarray:
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
        raise TypeError(
            "jobs_cmpt cost_options must be a str or ListLike type (List, Tuple, np.ndarray)"
        )

    return jobs_cmpt


def _standardize_simulation_cost_options_wjobs_cmpt(
    jobs_cmpt: np.ndarray, wjobs_cmpt: str | Numeric | ListLike, **kwargs
) -> (np.ndarray, str):
    if isinstance(wjobs_cmpt, str):
        if wjobs_cmpt == "mean":
            wjobs_cmpt = (
                np.ones(shape=jobs_cmpt.size, dtype=np.float32) / jobs_cmpt.size
            )

        elif wjobs_cmpt == "lquartile":
            wjobs_cmpt = np.ones(shape=jobs_cmpt.size, dtype=np.float32) * np.float32(
                -0.25
            )

        elif wjobs_cmpt == "median":
            wjobs_cmpt = np.ones(shape=jobs_cmpt.size, dtype=np.float32) * np.float32(
                -0.5
            )

        elif wjobs_cmpt == "uquartile":
            wjobs_cmpt = np.ones(shape=jobs_cmpt.size, dtype=np.float32) * np.float32(
                -0.75
            )

        else:
            raise ValueError(
                f"Unknown alias '{wjobs_cmpt}' for wjobs_cmpt cost_options. Choices: {WEIGHT_ALIAS}"
            )

    elif isinstance(wjobs_cmpt, (int, float, list, tuple, np.ndarray)):
        wjobs_cmpt = np.array(wjobs_cmpt, ndmin=1, dtype=np.float32)

        if wjobs_cmpt.size != jobs_cmpt.size:
            raise ValueError(
                f"Inconsistent sizes between jobs_cmpt ({jobs_cmpt.size}) and wjobs_cmpt ({wjobs_cmpt.size}) cost_options"
            )

        for i, wjoc in enumerate(wjobs_cmpt):
            if wjoc < 0:
                raise ValueError(
                    f"Negative component {wjoc:f} at index {i} in wjobs_cmpt cost_options"
                )

    else:
        raise TypeError(
            "wjobs_cmpt cost_options must be a str or Numeric type (int, float) or ListLike type (List, Tuple, np.ndarray)"
        )

    return wjobs_cmpt


def _standardize_simulation_cost_options_wjreg(wjreg: Numeric, **kwargs) -> float:
    if isinstance(wjreg, (int, float)):
        wjreg = float(wjreg)
        if wjreg < 0:
            raise ValueError("wjreg cost_options must be greater than or equal to 0")
    else:
        raise TypeError("wjreg cost_options must be of Numeric type (int, float)")

    return wjreg


def _standardize_simulation_cost_options_jreg_cmpt(
    wjreg: float, jreg_cmpt: str | ListLike, **kwargs
) -> np.ndarray:
    if wjreg > 0:
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
            raise TypeError(
                "jreg_cmpt cost_options must be a str or ListLike type (List, Tuple, np.ndarray)"
            )
    else:
        jreg_cmpt = np.empty(shape=0)

    return jreg_cmpt


def _standardize_simulation_cost_options_wjreg_cmpt(
    jreg_cmpt: np.ndarray, wjreg_cmpt: str | Numeric | ListLike, **kwargs
) -> (np.ndarray, str):
    if isinstance(wjreg_cmpt, str):
        if wjreg_cmpt == "mean":
            wjreg_cmpt = (
                np.ones(shape=jreg_cmpt.size, dtype=np.float32) / jreg_cmpt.size
            )

        elif wjreg_cmpt == "lquartile":
            wjreg_cmpt = np.ones(shape=jreg_cmpt.size, dtype=np.float32) * np.float32(
                -0.25
            )

        elif wjreg_cmpt == "median":
            wjreg_cmpt = np.ones(shape=jreg_cmpt.size, dtype=np.float32) * np.float32(
                -0.5
            )

        elif wjreg_cmpt == "uquartile":
            wjreg_cmpt = np.ones(shape=jreg_cmpt.size, dtype=np.float32) * np.float32(
                -0.75
            )

        else:
            raise ValueError(
                f"Unknown alias '{wjreg_cmpt}' for wjreg_cmpt cost_options. Choices: {WEIGHT_ALIAS}"
            )

    elif isinstance(wjreg_cmpt, (int, float, list, tuple, np.ndarray)):
        wjreg_cmpt = np.array(wjreg_cmpt, ndmin=1, dtype=np.float32)

        if wjreg_cmpt.size != jreg_cmpt.size:
            raise ValueError(
                f"Inconsistent sizes between jreg_cmpt ({jreg_cmpt.size}) and wjreg_cmpt ({wjreg_cmpt.size}) cost_options"
            )

        for i, wjrc in enumerate(wjreg_cmpt):
            if wjrc < 0:
                raise ValueError(
                    f"Negative component {wjrc:f} at index {i} in wjreg_cmpt cost_options"
                )

    else:
        raise TypeError(
            "wjreg_cmpt cost_options must be a str or Numeric type (int, float) or ListLike type (List, Tuple, np.ndarray)"
        )

    return wjreg_cmpt


def _standardize_simulation_cost_options_gauge(
    model: Model, gauge: str | ListLike, **kwargs
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
                f"Unknown alias or gauge code '{gauge}' for gauge cost_options. Choices: {GAUGE_ALIAS + list(model.mesh.code)}"
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
        raise TypeError(
            "gauge cost_options must be a str or ListLike type (List, Tuple, np.ndarray)"
        )

    return gauge


def _standardize_simulation_cost_options_wgauge(
    gauge: np.ndarray, wgauge: str | Numeric | ListLike, **kwargs
) -> (np.ndarray, str):
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
            raise ValueError(
                f"Unknown alias '{wgauge}' for wgauge cost_options. Choices: {WEIGHT_ALIAS}"
            )

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
            "wgauge cost_options must be a str or Numeric type (int, float) or ListLike type (List, Tuple, np.ndarray)"
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

            except:
                raise ValueError(
                    f"end_warmup '{end_warmup}' cost_options is an invalid date"
                )

        elif isinstance(end_warmup, pd.Timestamp):
            pass

        else:
            raise TypeError(
                f"end_warmup cost_options must be str or pandas.Timestamp object"
            )

        if (end_warmup - st).total_seconds() < 0 or (
            et - end_warmup
        ).total_seconds() < 0:
            raise ValueError(
                f"end_warmup '{end_warmup}' cost_options must be between start time '{st}' and end time '{et}'"
            )

    return end_warmup


def _standardize_simulation_cost_options(
    model: Model, cost_options: dict | None
) -> dict:
    if cost_options is None:
        cost_options = DEFAULT_SIMULATION_COST_OPTIONS.copy()

    else:
        if isinstance(cost_options, dict):
            pop_keys = []
            for key, value in cost_options.items():
                if key not in DEFAULT_SIMULATION_COST_OPTIONS.keys():
                    pop_keys.append(key)
                    warnings.warn(
                        f"Unknown cost_options key '{key}'. Choices: {list(DEFAULT_SIMULATION_COST_OPTIONS.keys())}"
                    )

            for key in pop_keys:
                cost_options.pop(key)

        else:
            raise TypeError("cost_options argument must be a dictionary")

    for key, value in DEFAULT_SIMULATION_COST_OPTIONS.items():
        cost_options.setdefault(key, value)
        func = eval(f"_standardize_simulation_cost_options_{key}")
        cost_options[key] = func(model=model, **cost_options)

    return cost_options


def _standardize_simulation_common_options_ncpu(ncpu: Numeric) -> int:
    if isinstance(ncpu, (int, float)):
        ncpu = int(ncpu)
        if ncpu <= 0:
            raise ValueError("ncpu common_options must be greater than 0")
        elif ncpu > os.cpu_count():
            ncpu = os.cpu_count()
            warnings.warn(
                f"ncpu common_options cannot be greater than the number of CPU(s) on the machine. ncpu is set to {os.cpu_count()}"
            )
    else:
        raise TypeError("ncpu common_options must be of Numeric type (int, float)")

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
            for key, value in common_options.items():
                if key not in DEFAULT_SIMULATION_COMMON_OPTIONS.keys():
                    pop_keys.append(key)
                    warnings.warn(
                        f"Unknown common_options key '{key}': Choices: {list(DEFAULT_SIMULATION_COMMON_OPTIONS.keys())}"
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
            time_step = pd.date_range(start=st, end=et, freq=f"{int(model.setup.dt)}s")[
                :-1
            ]
        else:
            try:
                # % Pass to list to convert to pd.DatetimeIndex
                time_step = [pd.Timestamp(time_step)]

            except:
                raise ValueError(
                    f"time_step '{time_step}' return_options is an invalid date"
                )

    elif isinstance(time_step, pd.Timestamp):
        # % Pass to list to convert to pd.DatetimeIndex
        time_step = [time_step]

    elif isinstance(time_step, (list, tuple, np.ndarray)):
        time_step = list(time_step)
        for i, date in enumerate(time_step):
            try:
                time_step[i] = pd.Timestamp(str(date))
            except:
                raise ValueError(
                    f"Invalid date '{date}' at index {i} in time_step return_options"
                )
    elif isinstance(time_step, pd.DatetimeIndex):
        pass

    else:
        raise TypeError(
            "time_step return_options must be a str, pd.Timestamp, pd.DatetimeIndex or ListLike type (List, Tuple, np.ndarray)"
        )

    time_step = pd.DatetimeIndex(time_step)

    # % Check that all dates are inside Model start_time and end_time
    for i, date in enumerate(time_step):
        if (date - st).total_seconds() < 0 or (et - date).total_seconds() < 0:
            raise ValueError(
                f"date '{date}' at index {i} in time_step return_options must be between start time '{st}' and end time '{et}'"
            )

    return time_step


def _standardize_simulation_return_options(
    model: Model, func_name: str, return_options: dict | None
) -> dict:
    if return_options is None:
        return_options = DEFAULT_SIMULATION_RETURN_OPTIONS[func_name].copy()
    else:
        if isinstance(return_options, dict):
            pop_keys = []
            for key, value in return_options.items():
                if key not in DEFAULT_SIMULATION_RETURN_OPTIONS[func_name].keys():
                    pop_keys.append(key)
                    warnings.warn(
                        f"Unknown return_options key '{key}': Choices: {list(DEFAULT_SIMULATION_RETURN_OPTIONS[func_name].keys())}"
                    )

            for key in pop_keys:
                return_options.pop(key)

        else:
            raise TypeError("return_options argument must be a dictionary")

    for key, value in DEFAULT_SIMULATION_RETURN_OPTIONS[func_name].items():
        return_options.setdefault(key, value)
        if key == "time_step":
            return_options[key] = _standardize_simulation_return_options_time_step(
                model, return_options[key]
            )
        else:
            _standardize_simulation_return_options_bool(key, return_options[key])

    return return_options


def _standardize_simulation_parameters_feasibility(model: Model):
    for key in model.opr_parameters.keys:
        arr = model.get_opr_parameters(key)
        l, u = FEASIBLE_OPR_PARAMETERS[key]
        l_arr = np.min(arr)
        u_arr = np.max(arr)

        if l_arr <= l or u_arr >= u:
            raise ValueError(
                f"Invalid value for model opr_parameter '{key}'. Opr_parameter domain [{l_arr}, {u_arr}] is not included in the feasible domain ]{l}, {u}["
            )

    for key in model.opr_initial_states.keys:
        arr = model.get_opr_initial_states(key)
        l, u = FEASIBLE_OPR_INITIAL_STATES[key]
        l_arr = np.min(arr)
        u_arr = np.max(arr)

        if l_arr <= l or u_arr >= u:
            raise ValueError(
                f"Invalid value for model opr_initial_states '{key}'. Opr_initial_state domain [{l_arr}, {u_arr}] is not included in the feasible domain ]{l}, {u}["
            )


def _standardize_simulation_optimize_options_finalize(
    model: Model, mapping: str, optimizer: str, optimize_options: dict
) -> dict:
    optimize_options["mapping"] = mapping
    optimize_options["optimizer"] = optimizer

    descriptor_present = "descriptor" in optimize_options.keys()

    # % Handle parameters
    # % opr parameters
    optimize_options["opr_parameters"] = np.zeros(shape=model.setup.nop, dtype=np.int32)
    optimize_options["l_opr_parameters"] = np.zeros(
        shape=model.setup.nop, dtype=np.float32
    )
    optimize_options["u_opr_parameters"] = np.zeros(
        shape=model.setup.nop, dtype=np.float32
    )

    if descriptor_present:
        optimize_options["opr_parameters_descriptor"] = np.zeros(
            shape=(model.setup.nd, model.setup.nop), dtype=np.int32
        )

    for i, key in enumerate(model.opr_parameters.keys):
        if key in optimize_options["parameters"]:
            optimize_options["opr_parameters"][i] = 1
            optimize_options["l_opr_parameters"][i] = optimize_options["bounds"][key][0]
            optimize_options["u_opr_parameters"][i] = optimize_options["bounds"][key][1]

            if descriptor_present:
                for j, desc in enumerate(model.setup.descriptor_name):
                    if desc in optimize_options["descriptor"][key]:
                        optimize_options["opr_parameters_descriptor"][j, i] = 1

    # % opr initial states
    optimize_options["opr_initial_states"] = np.zeros(
        shape=model.setup.nos, dtype=np.int32
    )
    optimize_options["l_opr_initial_states"] = np.zeros(
        shape=model.setup.nos, dtype=np.float32
    )
    optimize_options["u_opr_initial_states"] = np.zeros(
        shape=model.setup.nos, dtype=np.float32
    )
    if descriptor_present:
        optimize_options["opr_initial_states_descriptor"] = np.zeros(
            shape=(model.setup.nd, model.setup.nos), dtype=np.int32
        )

    for i, key in enumerate(model.opr_initial_states.keys):
        if key in optimize_options["parameters"]:
            optimize_options["opr_initial_states"][i] = 1
            optimize_options["l_opr_initial_states"][i] = optimize_options["bounds"][
                key
            ][0]
            optimize_options["u_opr_initial_states"][i] = optimize_options["bounds"][
                key
            ][1]

            if descriptor_present:
                for j, desc in enumerate(model.setup.descriptor_name):
                    if desc in optimize_options["descriptor"][key]:
                        optimize_options["opr_initial_states_descriptor"][j, i] = 1

    return optimize_options


def _standardize_simulation_cost_options_finalize(
    model: Model, cost_options: dict
) -> dict:
    cost_options["variant"] = "cls"  # only appeared in Fortran

    cost_options["njoc"] = cost_options["jobs_cmpt"].size
    cost_options["njrc"] = cost_options["jreg_cmpt"].size

    if any(f.startswith("E") for f in cost_options["jobs_cmpt"]):
        info_event = _mask_event(model=model, **cost_options["event_seg"])
        cost_options["n_event"] = info_event["n"]
        cost_options["mask_event"] = info_event["mask"]

    # % Handle flags send to Fortran
    # % gauge and wgauge
    gauge = np.zeros(shape=model.mesh.ng, dtype=np.int32)
    wgauge = np.zeros(shape=model.mesh.ng, dtype=np.float32)

    wg = dict(zip(cost_options["gauge"], cost_options["wgauge"]))

    for i, gc in enumerate(model.mesh.code):
        if gc in cost_options["gauge"]:
            gauge[i] = 1
            wgauge[i] = wg[gc]

    cost_options["gauge"] = gauge
    cost_options["wgauge"] = wgauge

    # % end_warmup
    st = pd.Timestamp(model.setup.start_time)
    cost_options["end_warmup"] = int(
        (cost_options["end_warmup"] - st).total_seconds() / model.setup.dt
    )

    return cost_options


def _standardize_simulation_return_options_finalize(model: Model, return_options: dict):
    st = pd.Timestamp(model.setup.start_time)

    mask_time_step = np.zeros(shape=model.setup.ntime_step, dtype=bool)

    for date in return_options["time_step"]:
        ind = int((date - st).total_seconds() / model.setup.dt)
        mask_time_step[ind] = True

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
            "fkeys": fkeys,
            "keys": keys,
        }
    )

    pop_keys = [
        k
        for k in return_options.keys()
        if k not in ["nmts", "mask_time_step", "time_step", "fkeys", "keys"]
    ]
    for key in pop_keys:
        return_options.pop(key)


def _standardize_default_optimize_options_args(
    mapping: str, optimizer: str | None
) -> dict:
    mapping = _standardize_simulation_mapping(mapping)

    optimizer = _standardize_simulation_optimizer(mapping, optimizer)

    return (mapping, optimizer)
