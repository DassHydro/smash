from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from smash._constant import (
    CONTROL_PRIOR_DISTRIBUTION,
    CONTROL_PRIOR_DISTRIBUTION_PARAMETERS,
    SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS,
)
from smash.core.model._build_model import _map_dict_to_fortran_derived_type
from smash.core.simulation._doc import (
    _bayesian_optimize_doc_appender,
    _multiple_optimize_doc_appender,
    _optimize_doc_appender,
    _smash_bayesian_optimize_doc_substitution,
    _smash_multiple_optimize_doc_substitution,
    _smash_optimize_doc_substitution,
)
from smash.core.simulation.optimize._standardize import (
    _standardize_multiple_optimize_args,
)
from smash.fcore._mw_forward import forward_run as wrap_forward_run
from smash.fcore._mw_optimize import (
    multiple_optimize as wrap_multiple_optimize,
)
from smash.fcore._mw_optimize import (
    optimize as wrap_optimize,
)
from smash.fcore._mwd_options import OptionsDT
from smash.fcore._mwd_parameters_manipulation import (
    parameters_to_control as wrap_parameters_to_control,
)
from smash.fcore._mwd_returns import ReturnsDT

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from smash.core.model.model import Model
    from smash.factory.net.net import Net
    from smash.factory.samples.samples import Samples


__all__ = [
    "MultipleOptimize",
    "Optimize",
    "BayesianOptimize",
    "multiple_optimize",
    "optimize",
    "bayesian_optimize",
]


class MultipleOptimize:
    """
    Represents multiple optimize result.

    Attributes
    ----------
    cost : `numpy.ndarray`
        An array of shape *(n,)* representing cost values from *n* simulations.

    q : `numpy.ndarray`
        An array of shape *(ng, ntime_step, n)* representing simulated discharges from *n* simulations.

    parameters : `dict[str, np.ndarray]`
        A dictionary containing optimized rainfall-runoff parameters and/or initial states.
        Each key represents an array of shape *(nrow, ncol, n)* corresponding to a specific rainfall-runoff
        parameter or initial state.

    See Also
    --------
    multiple_optimize : Run multiple optimization processes with multiple sets of parameters (i.e. starting
        points), yielding multiple solutions.
    """

    def __init__(self, data: dict[str, NDArray[np.float32]] | None = None):
        if data is None:
            data = {}

        self.__dict__.update(data)

    def __repr__(self):
        dct = self.__dict__

        if dct.keys():
            m = max(map(len, list(dct.keys()))) + 1
            return "\n".join(
                [k.rjust(m) + ": " + repr(type(v)) for k, v in sorted(dct.items()) if not k.startswith("_")]
            )
        else:
            return self.__class__.__name__ + "()"


class Optimize:
    """
    Represents optimize optional results.

    Attributes
    ----------
    time_step : `pandas.DatetimeIndex`
        A list of length *n* containing the returned time steps.

    rr_states : `FortranDerivedTypeArray`
        A list of length *n* of `RR_StatesDT <fcore._mwd_rr_states.RR_StatesDT>` for each **time_step**.

    q_domain : `numpy.ndarray`
        An array of shape *(nrow, ncol, n)* representing simulated discharges on the domain for each
        **time_step**.

    iter_cost : `numpy.ndarray`
        An array of shape *(m,)* representing cost iteration values from *m* iterations.

    iter_projg : `numpy.ndarray`
        An array of shape *(m,)* representing infinity norm of the projected gardient iteration values from
        *m* iterations.

    control_vector : `numpy.ndarray`
        An array of shape *(k,)* representing the control vector at end of optimization.

    net : `Net <factory.Net>`
        The trained neural network.

    cost : `float`
        Cost value.

    jobs : `float`
        Cost observation component value.

    jreg : `float`
        Cost regularization component value.

    lcurve_wjreg : `dict[str, Any]`
        A dictionary containing the wjreg lcurve data. The elements are:

        wjreg_opt : `float`
            The optimal wjreg value.

        distance : `numpy.ndarray`
            An array of shape *(6,)* representing the L-Curve distance for each optimization cycle
            (the maximum distance corresponds to the optimal wjreg).

        cost : `numpy.ndarray`
            An array of shape *(6,)* representing the cost values for each optimization cycle.

        jobs : `numpy.ndarray`
            An array of shape *(6,)* representing the jobs values for each optimization cycle.

        jreg : `numpy.ndarray`
            An array of shape *(6,)* representing the jreg values for each optimization cycle.

        wjreg : `numpy.ndarray`
            An array of shape *(6,)* representing the wjreg values for each optimization cycle.

    Notes
    -----
    The object's available attributes depend on what is requested by the user during a call to
    `optimize` in **return_options**.

    See Also
    --------
    smash.optimize : Model assimilation using numerical optimization algorithms.
    """

    def __init__(self, data: dict[str, Any] | None = None):
        if data is None:
            data = {}

        self.__dict__.update(data)

    def __repr__(self):
        dct = self.__dict__

        if dct.keys():
            m = max(map(len, list(dct.keys()))) + 1
            return "\n".join(
                [k.rjust(m) + ": " + repr(type(v)) for k, v in sorted(dct.items()) if not k.startswith("_")]
            )
        else:
            return self.__class__.__name__ + "()"


class BayesianOptimize:
    """
    Represents bayesian optimize optional results.

    Attributes
    ----------
    time_step : `pandas.DatetimeIndex`
        A list of length *n* containing the returned time steps.

    rr_states : `FortranDerivedTypeArray`
        A list of length *n* of `RR_StatesDT <fcore._mwd_rr_states.RR_StatesDT>` for each **time_step**.

    q_domain : `numpy.ndarray`
        An array of shape *(nrow, ncol, n)* representing simulated discharges on the domain for each
        **time_step**.

    iter_cost : `numpy.ndarray`
        An array of shape *(m,)* representing cost iteration values from *m* iterations.

    iter_projg : `numpy.ndarray`
        An array of shape *(m,)* representing infinity norm of the projected gardient iteration values from
        *m* iterations.

    control_vector : `numpy.ndarray`
        An array of shape *(k,)* representing the control vector at end of optimization.

    cost : `float`
        Cost value.

    log_lkh : `float`
        Log likelihood component value.

    log_prior : `float`
        Log prior component value.

    log_h : `float`
        Log h component value.

    serr_mu : `numpy.ndarray`
        An array of shape *(ng, ntime_step)* representing the mean of structural errors for each gauge and
        each **time_step**.

    serr_sigma : `numpy.ndarray`
        An array of shape *(ng, ntime_step)* representing the standard deviation of structural errors for
        each gauge and each **time_step**.

    Notes
    -----
    The object's available attributes depend on what is requested by the user during a call to
    `bayesian_optimize` in **return_options**.

    See Also
    --------
    smash.bayesian_optimize : Model bayesian assimilation using numerical optimization algorithms.
    """

    def __init__(self, data: dict[str, Any] | None = None):
        if data is None:
            data = {}

        self.__dict__.update(data)

    def __repr__(self):
        dct = self.__dict__

        if dct.keys():
            m = max(map(len, list(dct.keys()))) + 1
            return "\n".join(
                [k.rjust(m) + ": " + repr(type(v)) for k, v in sorted(dct.items()) if not k.startswith("_")]
            )
        else:
            return self.__class__.__name__ + "()"


def _get_control_info(
    model: Model,
    mapping: str,
    optimizer: str,
    optimize_options: dict,
    cost_options: dict,
) -> dict:
    wrap_options = OptionsDT(
        model.setup,
        model.mesh,
        cost_options["njoc"],
        cost_options["njrc"],
    )

    # % Map optimize_options dict to derived type
    _map_dict_to_fortran_derived_type(optimize_options, wrap_options.optimize)

    # % Map cost_options dict to derived type
    _map_dict_to_fortran_derived_type(cost_options, wrap_options.cost)

    wrap_parameters_to_control(model.setup, model.mesh, model._input_data, model._parameters, wrap_options)

    ret = {}
    for attr in dir(model._parameters.control):
        if attr.startswith("_"):
            continue
        value = getattr(model._parameters.control, attr)
        if callable(value):
            continue
        if hasattr(value, "copy"):
            value = value.copy()
        ret[attr] = value

    # Manually dealloc the control
    model._parameters.control.dealloc()

    return ret


def _get_fast_wjreg(model: Model, options: OptionsDT, returns: ReturnsDT) -> float:
    if options.comm.verbose:
        print(f"{' '*4}FAST WJREG CYCLE 1")

    # % Activate returns flags
    for flag in ["cost", "jobs", "jreg"]:
        setattr(returns, flag + "_flag", True)

    wrap_forward_run(
        model.setup,
        model.mesh,
        model._input_data,
        model._parameters,
        model._output,
        options,
        returns,
    )
    jobs0 = returns.jobs

    # Avoid to make a complete copy of model
    wparameters = model._parameters.copy()
    wrap_optimize(
        model.setup,
        model.mesh,
        model._input_data,
        wparameters,
        model._output,
        options,
        returns,
    )
    jobs = returns.jobs
    jreg = returns.jreg

    wjreg = (jobs0 - jobs) / jreg

    return wjreg


def _get_lcurve_wjreg_best(
    cost_arr: np.ndarray,
    jobs_arr: np.ndarray,
    jreg_arr: np.ndarray,
    wjreg_arr: np.ndarray,
) -> (np.ndarray, float):
    jobs_min = np.min(jobs_arr)
    jobs_max = np.max(jobs_arr)
    jreg_min = np.min(jreg_arr)
    jreg_max = np.max(jreg_arr)

    if (jobs_max - jobs_min) < 0 or (jreg_max - jreg_min) < 0:
        return np.empty(shape=0), 0.0

    max_distance = 0.0
    distance = np.zeros(shape=cost_arr.size)

    for i in range(cost_arr.size):
        lcurve_y = (jreg_arr[i] - jreg_min) / (jreg_max - jreg_min)
        lcurve_x = (jobs_max - jobs_arr[i]) / (jobs_max - jobs_min)
        # % Skip point above y = x
        if lcurve_y < lcurve_x:
            if jobs_arr[i] < jobs_max:
                hypot = np.hypot(lcurve_x, lcurve_y)
                alpha = np.pi * 0.25 - np.arccos(lcurve_x / hypot)
                distance[i] = hypot * np.sin(alpha)

            if distance[i] > max_distance:
                max_distance = distance[i]
                wjreg = wjreg_arr[i]

        else:
            distance[i] = np.nan

    return distance, wjreg


def _get_lcurve_wjreg(model: Model, options: OptionsDT, returns: ReturnsDT) -> (float, dict):
    if options.comm.verbose:
        print(f"{' '*4}LCURVE WJREG CYCLE 1")

    # % Activate returns flags
    for flag in ["cost", "jobs", "jreg"]:
        setattr(returns, flag + "_flag", True)

    wrap_forward_run(
        model.setup,
        model.mesh,
        model._input_data,
        model._parameters,
        model._output,
        options,
        returns,
    )
    jobs_max = returns.jobs

    # % Avoid to make a complete copy of model
    wparameters = model._parameters.copy()
    wrap_optimize(
        model.setup,
        model.mesh,
        model._input_data,
        wparameters,
        model._output,
        options,
        returns,
    )
    cost = returns.cost
    jobs_min = returns.jobs
    jreg_min = 0.0
    jreg_max = returns.jreg

    if (jobs_min / jobs_max) < 0.95 and (jreg_max - jreg_min) > 0.0:
        wjreg_fast = (jobs_max - jobs_min) / jreg_max
        log10_wjreg_fast = np.log10(wjreg_fast)
        wjreg_range = np.array(10 ** np.arange(log10_wjreg_fast - 0.66, log10_wjreg_fast + 0.67, 0.33))
    else:
        wjreg_range = np.empty(shape=0)

    nwjr = wjreg_range.size
    cost_arr = np.zeros(shape=nwjr + 1)
    cost_arr[0] = cost
    jobs_arr = np.zeros(shape=nwjr + 1)
    jobs_arr[0] = jobs_min
    jreg_arr = np.zeros(shape=nwjr + 1)
    jreg_arr[0] = jreg_max
    wjreg_arr = np.insert(wjreg_range, 0, 0.0)

    for i, wj in enumerate(wjreg_range):
        options.cost.wjreg = wj

        if options.comm.verbose:
            print(f"{' '*4}LCURVE WJREG CYCLE {i + 2}")

        wparameters = model._parameters.copy()
        wrap_optimize(
            model.setup,
            model.mesh,
            model._input_data,
            wparameters,
            model._output,
            options,
            returns,
        )

        cost_arr[i + 1] = returns.cost
        jobs_arr[i + 1] = returns.jobs
        jreg_arr[i + 1] = returns.jreg

    distance, wjreg = _get_lcurve_wjreg_best(cost_arr, jobs_arr, jreg_arr, wjreg_arr)

    lcurve = {
        "wjreg_opt": wjreg,
        "distance": distance,
        "cost": cost_arr,
        "jobs": jobs_arr,
        "jreg": jreg_arr,
        "wjreg": wjreg_arr,
    }

    return wjreg, lcurve


@_smash_optimize_doc_substitution
@_optimize_doc_appender
def optimize(
    model: Model,
    mapping: str = "uniform",
    optimizer: str | None = None,
    optimize_options: dict[str, Any] | None = None,
    cost_options: dict[str, Any] | None = None,
    common_options: dict[str, Any] | None = None,
    return_options: dict[str, Any] | None = None,
) -> Model | (Model, Optimize):
    wmodel = model.copy()

    ret_optimize = wmodel.optimize(
        mapping,
        optimizer,
        optimize_options,
        cost_options,
        common_options,
        return_options,
    )

    if ret_optimize is None:
        return wmodel
    else:
        return wmodel, ret_optimize


def _optimize(
    model: Model,
    mapping: str,
    optimizer: str,
    optimize_options: dict,
    cost_options: dict,
    common_options: dict,
    return_options: dict,
) -> Optimize | None:
    if common_options["verbose"]:
        print("</> Optimize")

    wrap_options = OptionsDT(
        model.setup,
        model.mesh,
        cost_options["njoc"],
        cost_options["njrc"],
    )

    wrap_returns = ReturnsDT(
        model.setup,
        model.mesh,
        return_options["nmts"],
        return_options["fkeys"],
    )

    # % Map optimize_options dict to derived type
    _map_dict_to_fortran_derived_type(optimize_options, wrap_options.optimize)

    # % Map cost_options dict to derived type
    _map_dict_to_fortran_derived_type(cost_options, wrap_options.cost)

    # % Map common_options dict to derived type
    _map_dict_to_fortran_derived_type(common_options, wrap_options.comm)

    # % Map return_options dict to derived type
    _map_dict_to_fortran_derived_type(return_options, wrap_returns)

    auto_wjreg = cost_options.get("auto_wjreg", None)

    if mapping == "ann":
        net = _ann_optimize(
            model,
            optimizer,
            optimize_options,
            common_options,
            wrap_options,
            wrap_returns,
        )

    else:
        if auto_wjreg == "fast":
            wrap_options.cost.wjreg = _get_fast_wjreg(model, wrap_options, wrap_returns)
            if wrap_options.comm.verbose:
                print(f"{' '*4}FAST WJREG LAST CYCLE. wjreg: {'{:.6f}'.format(wrap_options.cost.wjreg)}")
        elif auto_wjreg == "lcurve":
            wrap_options.cost.wjreg, lcurve_wjreg = _get_lcurve_wjreg(model, wrap_options, wrap_returns)
            if wrap_options.comm.verbose:
                print(f"{' '*4}LCURVE WJREG LAST CYCLE. wjreg: {'{:.6f}'.format(wrap_options.cost.wjreg)}")
        else:
            pass

        wrap_optimize(
            model.setup,
            model.mesh,
            model._input_data,
            model._parameters,
            model._output,
            wrap_options,
            wrap_returns,
        )

    fret = {}
    pyret = {}

    # % Fortran returns
    for key in return_options["keys"]:
        try:
            value = getattr(wrap_returns, key)
        except Exception:
            continue
        if hasattr(value, "copy"):
            value = value.copy()
        fret[key] = value

    # % ANN Python returns
    if mapping == "ann":
        if "net" in return_options["keys"]:
            pyret["net"] = net
        if "iter_cost" in return_options["keys"]:
            pyret["iter_cost"] = net.history["loss_train"]
        if "iter_projg" in return_options["keys"]:
            pyret["iter_projg"] = net.history["proj_grad"]

    # % L-curve wjreg return
    if auto_wjreg == "lcurve" and "lcurve_wjreg" in return_options["keys"]:
        pyret["lcurve_wjreg"] = lcurve_wjreg

    ret = {**fret, **pyret}
    if ret:
        # % Add time_step to the object
        if any(k in SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS for k in ret.keys()):
            ret["time_step"] = return_options["time_step"].copy()
        return Optimize(ret)


def _ann_optimize(
    model: Model,
    optimizer: str,
    optimize_options: dict,
    common_options: dict,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
) -> Net:
    # % Preprocessing input descriptors and normalization
    active_mask = np.where(model.mesh.active_cell == 1)

    l_desc = model._input_data.physio_data.l_descriptor
    u_desc = model._input_data.physio_data.u_descriptor

    desc = model._input_data.physio_data.descriptor.copy()
    desc = (desc - l_desc) / (u_desc - l_desc)  # normalize input descriptors

    # % Training the network
    x_train = desc[active_mask]

    net = optimize_options["net"]

    # % Change the mapping to trigger distributed control to get distributed gradients
    wrap_options.optimize.mapping = "distributed"

    net._fit_d2p(
        x_train,
        active_mask,
        model,
        wrap_options,
        wrap_returns,
        optimizer,
        optimize_options["parameters"],
        optimize_options["learning_rate"],
        optimize_options["random_state"],
        optimize_options["termination_crit"]["epochs"],
        optimize_options["termination_crit"]["early_stopping"],
        common_options["verbose"],
    )

    # % Manually deallocate control once fit_d2p done
    model._parameters.control.dealloc()

    # % Reset mapping to ann once fit_d2p done
    wrap_options.optimize.mapping = "ann"

    # % Predicting at active and inactive cells
    x = desc.reshape(-1, desc.shape[-1])
    y = net._predict(x)

    for i, name in enumerate(optimize_options["parameters"]):
        y_reshape = y[:, i].reshape(desc.shape[:2])

        if name in model.rr_parameters.keys:
            ind = np.argwhere(model.rr_parameters.keys == name).item()

            model.rr_parameters.values[..., ind] = y_reshape

        else:
            ind = np.argwhere(model.rr_initial_states.keys == name).item()

            model.rr_initial_states.values[..., ind] = y_reshape

    # % Forward run for updating final states
    wrap_forward_run(
        model.setup,
        model.mesh,
        model._input_data,
        model._parameters,
        model._output,
        wrap_options,
        wrap_returns,
    )

    return net


@_smash_multiple_optimize_doc_substitution
@_multiple_optimize_doc_appender
def multiple_optimize(
    model: Model,
    samples: Samples,
    mapping: str = "uniform",
    optimizer: str | None = None,
    optimize_options: dict[str, Any] | None = None,
    cost_options: dict[str, Any] | None = None,
    common_options: dict[str, Any] | None = None,
) -> MultipleOptimize:
    args_options = [deepcopy(arg) for arg in [optimize_options, cost_options, common_options]]

    args = _standardize_multiple_optimize_args(
        model,
        samples,
        mapping,
        optimizer,
        *args_options,
    )

    res = _multiple_optimize(model, *args)

    return MultipleOptimize(res)


def _multiple_optimize(
    model: Model,
    samples: Samples,
    mapping: str,
    optimizer: str,
    optimize_options: dict,
    cost_options: dict,
    common_options: dict,
) -> dict:
    if common_options["verbose"]:
        print("</> Multiple Optimize")

    wrap_options = OptionsDT(
        model.setup,
        model.mesh,
        cost_options["njoc"],
        cost_options["njrc"],
    )

    # % Map optimize_options dict to derived type
    _map_dict_to_fortran_derived_type(optimize_options, wrap_options.optimize)

    # % Map cost_options dict to derived type
    _map_dict_to_fortran_derived_type(cost_options, wrap_options.cost)

    # % Map common_options dict to derived type
    _map_dict_to_fortran_derived_type(common_options, wrap_options.comm)

    # % Generate samples info
    nv = samples._problem["num_vars"]
    samples_kind = np.zeros(shape=nv, dtype=np.int32, order="F")
    samples_ind = np.zeros(shape=nv, dtype=np.int32, order="F")

    for i, name in enumerate(samples._problem["names"]):
        if name in model._parameters.rr_parameters.keys:
            samples_kind[i] = 0
            # % Adding 1 because Fortran uses one based indexing
            samples_ind[i] = np.argwhere(model._parameters.rr_parameters.keys == name).item() + 1
        elif name in model._parameters.rr_initial_states.keys:
            samples_kind[i] = 1
            # % Adding 1 because Fortran uses one based indexing
            samples_ind[i] = np.argwhere(model._parameters.rr_initial_states.keys == name).item() + 1
        # % Should be unreachable
        else:
            pass

    # % Initialise results
    cost = np.zeros(shape=samples.n_sample, dtype=np.float32, order="F")
    q = np.zeros(
        shape=(*model.response_data.q.shape, samples.n_sample),
        dtype=np.float32,
        order="F",
    )
    # % Only work with grids (might be changed)
    parameters = np.zeros(
        shape=(
            *model.mesh.flwdir.shape,
            len(optimize_options["parameters"]),
            samples.n_sample,
        ),
        dtype=np.float32,
        order="F",
    )

    wrap_multiple_optimize(
        model.setup,
        model.mesh,
        model._input_data,
        model._parameters,
        model._output,
        wrap_options,
        samples.to_numpy(),
        samples_kind,
        samples_ind,
        cost,
        q,
        parameters,
    )

    # % Finalize parameters and samples for returns
    parameters = dict(
        zip(
            optimize_options["parameters"],
            np.transpose(parameters, (2, 0, 1, 3)),
        )
    )

    for sp in samples._problem["names"]:  # add uncalibrated parameters from samples to parameters
        if sp not in optimize_options["parameters"]:
            value = getattr(samples, sp)
            value = np.tile(value, (*model.mesh.flwdir.shape, 1))

            parameters.update({sp: value})

    samples_fnl = deepcopy(samples)  # make a deepcopy of samples (will be modified by setattr)

    for op in optimize_options["parameters"]:  # add calibrated paramters from parameters to samples
        if op not in samples._problem["names"]:
            if op in model.rr_parameters.keys:
                value = model.get_rr_parameters(op)[0, 0]

            elif op in model.rr_initial_states.keys:
                value = model.get_rr_initial_states(op)[0, 0]

            # % In case we have other kind of parameters. Should be unreachable.
            else:
                pass

            setattr(samples_fnl, op, value * np.ones(samples.n_sample))
            setattr(samples_fnl, "_dst_" + op, np.ones(samples.n_sample))

    return {
        "cost": cost,
        "q": q,
        "parameters": parameters,
        "_samples": samples_fnl,
        "_cost_options": cost_options,
    }


def _handle_bayesian_optimize_control_prior(model: Model, control_prior: dict, options: OptionsDT):
    wrap_parameters_to_control(model.setup, model.mesh, model._input_data, model._parameters, options)

    if control_prior is None:
        control_prior = {}

    elif isinstance(control_prior, dict):
        for key, value in control_prior.items():
            if key not in model._parameters.control.name:
                raise ValueError(
                    f"Unknown control name '{key}' in control_prior cost_options. "
                    f"Choices: {list(model._parameters.control.name)}"
                )
            else:
                if isinstance(value, (list, tuple, np.ndarray)):
                    if value[0] not in CONTROL_PRIOR_DISTRIBUTION:
                        raise ValueError(
                            f"Unknown distribution '{value[0]}' for key '{key}' in control_prior "
                            f"cost_options. Choices: {CONTROL_PRIOR_DISTRIBUTION}"
                        )
                    value[1] = np.array(value[1], dtype=np.float32)
                    if value[1].size != CONTROL_PRIOR_DISTRIBUTION_PARAMETERS[value[0]]:
                        raise ValueError(
                            f"Invalid number of parameter(s) ({value[1].size}) for distribution '{value[0]}' "
                            f"for key '{key}' in control_prior cost_options. "
                            f"Expected: ({CONTROL_PRIOR_DISTRIBUTION_PARAMETERS[value[0]]})"
                        )
                else:
                    raise ValueError(
                        f"control_prior cost_options value for key '{key}' must be of ListLike (List, "
                        f"Tuple, np.ndarray)"
                    )
            control_prior[key] = {"dist": value[0], "par": value[1]}
    else:
        raise TypeError("control_prior cost_options must be a dictionary")

    for key in model._parameters.control.name:
        control_prior.setdefault(key, {"dist": "FlatPrior", "par": np.empty(shape=0)})

    # % allocate control prior
    npar = np.array([p["par"].size for p in control_prior.values()], dtype=np.int32)
    options.cost.alloc_control_prior(model._parameters.control.n, npar)

    # % map control prior dict to derived type array
    for i, prior in enumerate(control_prior.values()):
        _map_dict_to_fortran_derived_type(prior, options.cost.control_prior[i])


@_smash_bayesian_optimize_doc_substitution
@_bayesian_optimize_doc_appender
def bayesian_optimize(
    model: Model,
    mapping: str = "uniform",
    optimizer: str | None = None,
    optimize_options: dict[str, Any] | None = None,
    cost_options: dict[str, Any] | None = None,
    common_options: dict[str, Any] | None = None,
    return_options: dict[str, Any] | None = None,
) -> Model | (Model, BayesianOptimize):
    wmodel = model.copy()

    ret_bayesian_optimize = wmodel.bayesian_optimize(
        mapping,
        optimizer,
        optimize_options,
        cost_options,
        common_options,
        return_options,
    )

    if ret_bayesian_optimize is None:
        return wmodel
    else:
        return wmodel, ret_bayesian_optimize


def _bayesian_optimize(
    model: Model,
    mapping: str,
    optimizer: str,
    optimize_options: dict,
    cost_options: dict,
    common_options: dict,
    return_options: dict,
) -> BayesianOptimize | None:
    if common_options["verbose"]:
        print("</> Bayesian Optimize")

    wrap_options = OptionsDT(
        model.setup,
        model.mesh,
        cost_options["njoc"],
        cost_options["njrc"],
    )

    wrap_returns = ReturnsDT(
        model.setup,
        model.mesh,
        return_options["nmts"],
        return_options["fkeys"],
    )

    # % Map optimize_options dict to derived type
    _map_dict_to_fortran_derived_type(optimize_options, wrap_options.optimize)

    # % Map cost_options dict to derived type
    # % Control prior handled after
    _map_dict_to_fortran_derived_type(cost_options, wrap_options.cost, skip=["control_prior"])

    # % Map common_options dict to derived type
    _map_dict_to_fortran_derived_type(common_options, wrap_options.comm)

    # % Map return_options dict to derived type
    _map_dict_to_fortran_derived_type(return_options, wrap_returns)

    # % Control prior check
    _handle_bayesian_optimize_control_prior(model, cost_options["control_prior"], wrap_options)

    wrap_optimize(
        model.setup,
        model.mesh,
        model._input_data,
        model._parameters,
        model._output,
        wrap_options,
        wrap_returns,
    )

    fret = {}
    pyret = {}

    # % Fortran returns
    for key in return_options["keys"]:
        try:
            value = getattr(wrap_returns, key)
        except Exception:
            continue
        if hasattr(value, "copy"):
            value = value.copy()
        fret[key] = value

    ret = {**fret, **pyret}
    if ret:
        # % Add time_step to the object
        if any(k in SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS for k in ret.keys()):
            ret["time_step"] = return_options["time_step"].copy()
        return BayesianOptimize(ret)
