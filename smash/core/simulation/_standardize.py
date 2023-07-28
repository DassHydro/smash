from __future__ import annotations

from smash._constant import (
    STRUCTURE_OPR_PARAMETERS,
    STRUCTURE_OPR_STATES,
    OPTIMIZABLE_OPR_PARAMETERS,
    OPTIMIZABLE_OPR_INITIAL_STATES,
    FEASIBLE_OPR_PARAMETERS,
    FEASIBLE_OPR_INITIAL_STATES,
    MAPPING,
    COST_VARIANT,
    MAPPING_OPTIMIZER,
    OPTIMIZER_CONTROL_TFM,
    DEFAULT_SIMULATION_OPTIMIZE_OPTIONS,
    DEFAULT_SIMULATION_COMMON_OPTIONS,
    DEFAULT_SIMULATION_COST_OPTIONS,
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

import numpy as np
import pandas as pd
import warnings
import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash._typing import Numeric, ListLike
    from smash.core.model.model import Model
    from smash.factory.net.net import Net


def _standardize_simulation_mapping(mapping: str) -> str:
    if isinstance(mapping, str):
        if mapping.lower() not in MAPPING:
            raise ValueError(f"Unknown mapping '{mapping}'. Choices: {MAPPING}")
    else:
        raise TypeError("mapping argument must be a str")

    return mapping.lower()


def _standardize_simulation_cost_variant(cost_variant: str) -> str:
    if isinstance(cost_variant, str):
        if cost_variant.lower() not in COST_VARIANT:
            raise ValueError(
                f"Unknown cost_variant '{cost_variant}'. Choices: {COST_VARIANT}"
            )
    else:
        raise TypeError("cost_variant argument must be a str")

    return cost_variant.lower()


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
        # % In case we have other kind of parameters
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
) -> str:
    # % "..." will not trigger any transformation in Fortran
    if control_tfm is None:
        control_tfm = "..."
    else:
        if isinstance(control_tfm, str):
            if control_tfm.lower() not in OPTIMIZER_CONTROL_TFM[optimizer]:
                raise ValueError(
                    f"Unknown transformation '{control_tfm}' in control_tfm optimize_options. Choices: {OPTIMIZER_CONTROL_TFM[optimizer]}"
                )
        else:
            TypeError("control_tfm optimize_options must be a str")

    return control_tfm


# % TODO: Add check for ANN (same descriptors for all optimized parameters)
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


# % TODO: TH add standardize for net
def _standardize_simulation_optimize_options_net(
    model: Model, net: Net | None, bounds: dict, **kwargs
) -> Net:
    return net


def _standardize_simulation_optimize_options_maxiter(maxiter: Numeric, **kwargs) -> int:
    if isinstance(maxiter, (int, float)):
        maxiter = int(maxiter)
        if maxiter < 0:
            raise ValueError(
                "maxiter optimize_options must be greater than or equal to 0"
            )
    else:
        raise TypeError("maxiter optimize_options must be of Numeric type (int, float)")

    return maxiter


def _standardize_simulation_optimize_options_factr(factr: Numeric, **kwargs) -> float:
    if isinstance(factr, (int, float)):
        factr = float(factr)
        if factr <= 0:
            raise ValueError("factr optimize_options must be greater than 0")
    else:
        raise TypeError("factr optimize_options must be of Numeric type (int, float)")

    return factr


def _standardize_simulation_optimize_options_pgtol(pgtol: Numeric, **kwargs) -> float:
    if isinstance(pgtol, (int, float)):
        pgtol = float(pgtol)
        if pgtol <= 0:
            raise ValueError("pgtol optimize_options must be greater than 0")
    else:
        raise TypeError("pgtol optimize_options must be of Numeric type (int, float)")

    return pgtol


# % TODO: TH check this
def _standardize_simulation_optimize_options_epochs(epochs: Numeric, **kwargs) -> float:
    if isinstance(epochs, (int, float)):
        epochs = int(epochs)
        if epochs < 0:
            raise ValueError(
                "epochs optimize_options must be greater than or equal to 0"
            )
    else:
        raise TypeError("epochs optimize_options must be of Numeric type (int, float)")

    return epochs


# % TODO: TH check this
def _standardize_simulation_optimize_options_lerning_rate(
    lerning_rate: Numeric, **kwargs
) -> float:
    if isinstance(lerning_rate, (int, float)):
        lerning_rate = float(lerning_rate)
        if lerning_rate < 0:
            raise ValueError("lerning_rate optimize_options must be greater than 0")
    else:
        raise TypeError(
            "lerning_rate optimize_options must be of Numeric type (int, float)"
        )

    return lerning_rate


# % TODO: TH check this
def _standardize_simulation_optimize_options_early_stopping(
    early_stopping: bool, **kwargs
) -> float:
    if isinstance(early_stopping, bool):
        pass
    else:
        raise TypeError("early_stopping optimize_options must be of a boolean")

    return early_stopping


def _standardize_simulation_optimize_options(
    model: Model, mapping: str, optimizer: str, optimize_options: dict | None
) -> dict:
    if optimize_options is None:
        optimize_options = DEFAULT_SIMULATION_OPTIMIZE_OPTIONS[
            (mapping, optimizer)
        ].copy()

    else:
        if isinstance(optimize_options, dict):
            pop_keys = []
            for key, value in optimize_options.items():
                if (
                    key
                    not in DEFAULT_SIMULATION_OPTIMIZE_OPTIONS[
                        (mapping, optimizer)
                    ].keys()
                ):
                    pop_keys.append(key)
                    warnings.warn(
                        f"Unknown optimize_options key '{key}' for mapping '{mapping}' and optimizer '{optimizer}'. Choices: {list(DEFAULT_SIMULATION_OPTIMIZE_OPTIONS[(mapping, optimizer)].keys())}"
                    )

            for key in pop_keys:
                optimize_options.pop(key)

        else:
            raise TypeError("optimize_options argument must be a dictionary")

    for key, value in DEFAULT_SIMULATION_OPTIMIZE_OPTIONS[(mapping, optimizer)].items():
        optimize_options.setdefault(key, value)
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
) -> np.ndarray:
    if isinstance(wjobs_cmpt, str):
        if wjobs_cmpt == "mean":
            wjobs_cmpt = (
                np.ones(shape=jobs_cmpt.size, dtype=np.float32) / jobs_cmpt.size
            )

        elif wjobs_cmpt == "median":
            wjobs_cmpt = np.ones(shape=jobs_cmpt.size, dtype=np.float32) * np.float(
                -0.5
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
    jreg_cmpt: str | ListLike | None, **kwargs
) -> np.ndarray:
    if jreg_cmpt is None:
        jreg_cmpt = np.empty(shape=0)
    else:
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

    return jreg_cmpt


def _standardize_simulation_cost_options_wjreg_cmpt(
    jreg_cmpt: np.ndarray, wjreg_cmpt: str | Numeric | ListLike, **kwargs
) -> np.ndarray:
    if isinstance(wjreg_cmpt, str):
        if wjreg_cmpt == "mean":
            wjreg_cmpt = (
                np.ones(shape=jreg_cmpt.size, dtype=np.float32) / jreg_cmpt.size
            )

        elif wjreg_cmpt == "median":
            wjreg_cmpt = np.ones(shape=jreg_cmpt.size, dtype=np.float32) * np.float(
                -0.5
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
            gauge = np.zeros(shape=model.mesh.ng, dtype=np.int32)
            for i, pos in enumerate(model.mesh.gauge_pos):
                if model.mesh.flwdst[tuple(pos)] == 0:
                    gauge[i] = 1

        elif gauge == "all":
            gauge = np.ones(shape=model.mesh.ng, dtype=np.int32)

        elif gauge in model.mesh.code:
            ind = np.argwhere(model.mesh.code == gauge).squeeze()
            gauge = np.zeros(shape=model.mesh.ng, dtype=np.int32)
            gauge[ind] = 1

        else:
            raise ValueError(
                f"Unknown alias or gauge code '{gauge}' for gauge cost_options. Choices: {GAUGE_ALIAS + list(model.mesh.code)}"
            )

    elif isinstance(gauge, (list, tuple, np.ndarray)):
        gauge_bak = np.array(gauge, ndmin=1)
        gauge = np.zeros(shape=model.mesh.ng, dtype=np.int32)

        for i, ggc in enumerate(gauge_bak):
            if ggc in model.mesh.code:
                gauge[i] = 1
            else:
                raise ValueError(
                    f"Unknown gauge code '{ggc}' at index {i} in gauge cost_options. Choices: {list(model.mesh.code)}"
                )

    else:
        raise TypeError(
            "gauge cost_options must be a str or ListLike type (List, Tuple, np.ndarray)"
        )

    return gauge


def _standardize_simulation_cost_options_wgauge(
    gauge: np.ndarray, model: Model, wgauge: str | Numeric | ListLike, **kwargs
) -> np.ndarray:
    if isinstance(wgauge, str):
        if wgauge == "mean":
            wgauge = np.zeros(shape=model.mesh.ng, dtype=np.float32)
            wgauge = np.where(gauge == 1, 1 / np.sum(gauge), wgauge)

        elif wgauge == "median":
            wgauge = np.zeros(shape=model.mesh.ng, dtype=np.float32)
            wgauge = np.where(gauge == 1, np.float(-0.5), wgauge)

        else:
            raise ValueError(
                f"Unknown alias '{wgauge}' for wgauge cost_options. Choices: {WEIGHT_ALIAS}"
            )

    elif isinstance(wgauge, (int, float, list, tuple, np.ndarray)):
        wgauge_bak = np.array(wgauge, ndmin=1)
        wgauge = np.zeros(shape=model.mesh.ng, dtype=np.float32)

        if wgauge_bak.size != np.sum(gauge):
            raise ValueError(
                f"Inconsistent sizes between wgauge ({wgauge_bak.size}) and gauge ({np.sum(gauge)}) cost_options"
            )

        for i, wggc in enumerate(wgauge_bak):
            if wggc < 0:
                raise ValueError(
                    f"Negative component {wggc:f} at index {i} in wgauge cost_options"
                )
            if gauge[i] == 1:
                wgauge[i] = wggc

    else:
        raise TypeError(
            "wgauge cost_options must be a str or Numeric type (int, float) or ListLike type (List, Tuple, np.ndarray)"
        )

    return wgauge


def _standardize_simulation_cost_options_event_seg(
    event_seg: dict | None, **kwargs
) -> dict:
    if event_seg is None:
        event_seg = {}

    else:
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
            raise TypeError("event_seg cost_options must be a dictionary")

    return event_seg


def _standardize_simulation_cost_options_end_warmup(
    model: Model, end_warmup: str | pd.Timestamp | None, **kwargs
) -> int:
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

    end_warmup = int((end_warmup - st).total_seconds() / model.setup.dt)

    return end_warmup


def _standardize_simulation_cost_options(
    model: Model, cost_variant: str, cost_options: dict | None
) -> dict:
    if cost_options is None:
        cost_options = DEFAULT_SIMULATION_COST_OPTIONS[cost_variant].copy()

    else:
        if isinstance(cost_options, dict):
            pop_keys = []
            for key, value in cost_options.items():
                if key not in DEFAULT_SIMULATION_COST_OPTIONS[cost_variant].keys():
                    pop_keys.append(key)
                    warnings.warn(
                        f"Unknown cost_options key '{key}' for cost_variant '{cost_variant}'. Choices: {list(DEFAULT_SIMULATION_COST_OPTIONS[cost_variant].keys())}"
                    )

            for key in pop_keys:
                cost_options.pop(key)

        else:
            raise TypeError("cost_options argument must be a dictionary")

    for key, value in DEFAULT_SIMULATION_COST_OPTIONS[cost_variant].items():
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


# % TODO: TH Implement mask_event function
def _standardize_simulation_cost_options_finalize(
    model: Model, cost_variant: str, cost_options: dict
) -> dict:
    cost_options["variant"] = cost_variant
    cost_options["njoc"] = cost_options["jobs_cmpt"].size
    cost_options["njrc"] = cost_options["jreg_cmpt"].size
    # ~ cost_options["mask_event"] = _mask_event(**cost_options["event_seg"])

    return cost_options
