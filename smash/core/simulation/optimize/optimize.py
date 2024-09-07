from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from smash._constant import (
    CONTROL_PRIOR_DISTRIBUTION,
    CONTROL_PRIOR_DISTRIBUTION_PARAMETERS,
    SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS,
    STRUCTURE_RR_INTERNAL_FLUXES,
)
from smash.core.model._build_model import _map_dict_to_fortran_derived_type
from smash.core.simulation._doc import (
    _bayesian_optimize_doc_appender,
    _optimize_doc_appender,
    _smash_bayesian_optimize_doc_substitution,
    _smash_optimize_doc_substitution,
)
from smash.factory.net._loss import _inf_norm  # this import will be removed when all optimizers are combined
from smash.fcore._mw_forward import forward_run as wrap_forward_run
from smash.fcore._mw_forward import forward_run_b as wrap_forward_run_b
from smash.fcore._mwd_options import OptionsDT
from smash.fcore._mwd_parameters_manipulation import (
    parameters_to_control as wrap_parameters_to_control,
)
from smash.fcore._mwd_returns import ReturnsDT

if TYPE_CHECKING:
    from typing import Any

    from scipy.optimize import OptimizeResult as scipy_OptimizeResult

    from smash.core.model.model import Model
    from smash.factory.net.net import Net
    from smash.fcore._mwd_parameters import ParametersDT


__all__ = [
    "Optimize",
    "BayesianOptimize",
    "optimize",
    "bayesian_optimize",
]


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

    internal_fluxes: `dict[str, numpy.ndarray]`
        A dictionary where keys are the names of the internal fluxes and the values are array of
        shape *(nrow, ncol, n)* representing an internal flux on the domain for each **time_step**.

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

    internal_fluxes: `dict[str, numpy.ndarray]`
        A dictionary where keys are the names of the internal fluxes and the values are array of
        shape *(nrow, ncol, n)* representing an internal flux on the domain for each **time_step**.

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


class _OptimizeCallback:
    # % Private callback class for external optimizer i.e. L-BFGS-B from scipy
    def __init__(self, verbose: bool):
        self.verbose = verbose

        self.iterations = 0

        self.nfg = 1
        self.count_nfg = 0

        self.iter_cost = np.array([])

        self.iter_projg = np.array([])
        self.projg = None

    def callback(self, intermediate_result: scipy_OptimizeResult):
        # % intermediate_result is required by callback function in scipy
        if self.verbose:
            print(
                f"    At iterate{str(self.iterations).rjust(7)}    nfg = {str(self.nfg).rjust(4)}"
                f"    J = {self.iter_cost[-1]:14.6f}    |proj g| = {self.projg:14.6f}"
            )

        self.iterations += 1

        self.nfg = self.count_nfg

        self.iter_cost = np.append(self.iter_cost, intermediate_result.fun)

        self.iter_projg = np.append(self.iter_projg, self.projg)
        self.projg = self.tmp_projg

    def termination(self, final_result: scipy_OptimizeResult):
        if self.verbose:
            print(
                f"    At iterate{str(self.iterations).rjust(7)}    nfg = {str(self.nfg).rjust(4)}"
                f"    J = {final_result.fun:14.6f}    |proj g| = {self.projg:14.6f}"
            )
            print(f"    {final_result.message}")

        self.iter_projg = np.append(self.iter_projg, self.projg)


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


def _get_fast_wjreg(model: Model, options: OptionsDT, returns: ReturnsDT, return_options: dict) -> float:
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
    _apply_optimizer(model, wparameters, options, returns, return_options)

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


def _get_lcurve_wjreg(
    model: Model, options: OptionsDT, returns: ReturnsDT, return_options: dict
) -> (float, dict):
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
    _apply_optimizer(model, wparameters, options, returns, return_options)

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
        _apply_optimizer(model, wparameters, options, returns, return_options)

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
        )  # % TODO TH: this function will be merged into _apply_optimizer

        pyret = {}

    else:
        if auto_wjreg == "fast":
            wrap_options.cost.wjreg = _get_fast_wjreg(model, wrap_options, wrap_returns, return_options)
            if wrap_options.comm.verbose:
                print(f"{' '*4}FAST WJREG LAST CYCLE. wjreg: {'{:.6f}'.format(wrap_options.cost.wjreg)}")
        elif auto_wjreg == "lcurve":
            wrap_options.cost.wjreg, lcurve_wjreg = _get_lcurve_wjreg(
                model, wrap_options, wrap_returns, return_options
            )
            if wrap_options.comm.verbose:
                print(f"{' '*4}LCURVE WJREG LAST CYCLE. wjreg: {'{:.6f}'.format(wrap_options.cost.wjreg)}")
        else:
            pass

        pyret = _apply_optimizer(model, model._parameters, wrap_options, wrap_returns, return_options)

    fret = {}

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
        if "internal_fluxes" in ret:
            ret["internal_fluxes"] = {
                key: ret["internal_fluxes"][..., i]
                for i, key in enumerate(STRUCTURE_RR_INTERNAL_FLUXES[model.setup.structure])
            }

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
    l_desc = model._input_data.physio_data.l_descriptor
    u_desc = model._input_data.physio_data.u_descriptor

    desc = model._input_data.physio_data.descriptor.copy()
    desc_shape = desc.shape
    desc = (desc - l_desc) / (u_desc - l_desc)  # normalize input descriptors

    # % Train regionalization network
    net = optimize_options["net"]

    if net.layers[0].layer_name() == "Dense":
        desc = desc.reshape(-1, desc_shape[-1])

    # % Change the mapping to trigger distributed control to get distributed gradients
    wrap_options.optimize.mapping = "distributed"

    net._fit_d2p(
        desc,
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

    # % Run a forward pass with net
    y = net._forward_pass(desc)

    if y.ndim < 3:
        y = y.reshape(desc_shape[:-1] + (-1,))

    for i, name in enumerate(optimize_options["parameters"]):
        if name in model.rr_parameters.keys:
            ind = np.argwhere(model.rr_parameters.keys == name).item()

            model.rr_parameters.values[..., ind] = y[..., i]

        elif name in model.rr_initial_states.keys:
            ind = np.argwhere(model.rr_initial_states.keys == name).item()

            model.rr_initial_states.values[..., ind] = y[..., i]

        else:  # nn_parameters excluded from descriptors-to-parameters mapping
            pass

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

    pyret = _apply_optimizer(model, model._parameters, wrap_options, wrap_returns, return_options)

    # % Fortran returns
    fret = {}
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
        if "internal_fluxes" in ret:
            ret["internal_fluxes"] = {
                key: ret["internal_fluxes"][..., i]
                for i, key in enumerate(STRUCTURE_RR_INTERNAL_FLUXES[model.setup.structure])
            }
        # % Add time_step to the object
        if any(k in SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS for k in ret.keys()):
            ret["time_step"] = return_options["time_step"].copy()
        return BayesianOptimize(ret)


def _apply_optimizer(
    model: Model,
    parameters: ParametersDT,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
    return_options: dict,
) -> dict:
    if wrap_options.optimize.optimizer == "sbs":
        ret = _sbs_optimize(model, parameters, wrap_options, wrap_returns, return_options)

    elif wrap_options.optimize.optimizer == "lbfgsb":
        ret = _lbfgsb_optimize(model, parameters, wrap_options, wrap_returns, return_options)

    else:  # % Machine learning optimizer (adam, adagrad, etc.)
        pass  # TODO TH: add ML opt

    # % Manually deallocate control
    parameters.control.dealloc()

    return ret


def _lbfgsb_optimize(
    model: Model,
    parameters: ParametersDT,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
    return_options: dict,
) -> dict:
    wrap_parameters_to_control(
        model.setup,
        model.mesh,
        model._input_data,
        parameters,
        wrap_options,
    )

    x0 = parameters.control.x.copy()

    # % Set None values for unbounded/semi-unbounded controls to pass to scipy l-bfgs-b
    # which requires None to identify unbounded values
    l_control = np.where(np.isin(parameters.control.nbd, [0, 3]), None, parameters.control.l)
    u_control = np.where(np.isin(parameters.control.nbd, [0, 1]), None, parameters.control.u)

    cb = _OptimizeCallback(wrap_options.comm.verbose)

    res_optimize = scipy_minimize(
        _gradient_based_optimize_problem,
        x0,
        args=(model, parameters, wrap_options, wrap_returns, cb),
        method="l-bfgs-b",
        jac=True,
        bounds=tuple(zip(l_control, u_control)),
        callback=cb.callback,
        options={
            "maxiter": wrap_options.optimize.maxiter,
            "ftol": 2.22e-16 * wrap_options.optimize.factr,
            "gtol": wrap_options.optimize.pgtol,
            "disp": False,  # TODO: change this with logger for multiple display levels
        },
    )

    cb.termination(res_optimize)

    # % Apply final control and forward run for updating final states
    setattr(parameters.control, "x", res_optimize["x"])

    wrap_forward_run(
        model.setup,
        model.mesh,
        model._input_data,
        parameters,
        model._output,
        wrap_options,
        wrap_returns,
    )

    ret = {}

    if "control_vector" in return_options["keys"]:
        ret["control_vector"] = parameters.control.x.copy()

    if "iter_cost" in return_options["keys"]:
        ret["iter_cost"] = cb.iter_cost

    if "iter_projg" in return_options["keys"]:
        ret["iter_projg"] = cb.iter_projg

    if "serr_mu" in return_options["keys"]:
        ret["serr_mu"] = model.get_serr_mu().copy()

    if "serr_sigma" in return_options["keys"]:
        ret["serr_sigma"] = model.get_serr_sigma().copy()

    return ret


def _sbs_optimize(
    model: Model,
    parameters: ParametersDT,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
    return_options: dict,
) -> dict:
    wrap_parameters_to_control(
        model.setup,
        model.mesh,
        model._input_data,
        parameters,
        wrap_options,
    )

    n = parameters.control.n

    sdx = np.zeros(n)

    y_wa = parameters.control.x.copy()
    z_wa = np.copy(y_wa)

    # % Set np.inf values for unbounded/semi-unbounded controls
    l_wa = np.where(np.isin(parameters.control.nbd, [0, 3]), -np.inf, parameters.control.l)
    u_wa = np.where(np.isin(parameters.control.nbd, [0, 1]), np.inf, parameters.control.u)

    wrap_forward_run(
        model.setup,
        model.mesh,
        model._input_data,
        parameters,
        model._output,
        wrap_options,
        wrap_returns,
    )

    gx = model._output.cost
    ga = gx
    clg = 0.7 ** (1 / n)
    ddx = 0.64
    dxn = ddx
    ia = iaa = iam = -1
    jfa = jfaa = 0
    nfg = 1

    ret = {}

    message = "STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT"

    if wrap_options.comm.verbose:
        print(
            f"    At iterate{str(0).rjust(7)}    nfg = {str(nfg).rjust(4)}"
            f"    J = {gx:14.6f}    ddx = {ddx:5.2f}"
        )

    if "iter_cost" in return_options["keys"]:
        ret["iter_cost"] = np.array([gx])

    for iter in range(1, wrap_options.optimize.maxiter * n + 1):
        if dxn > ddx:
            dxn = ddx
        if ddx > 2:
            ddx = dxn

        for i in range(n):
            x_wa = np.copy(y_wa)

            for j in range(1, 3):
                jf = 2 * j - 3
                if i == iaa and jf == -jfaa:
                    continue
                if y_wa[i] <= l_wa[i] and jf < 0:
                    continue
                if y_wa[i] >= u_wa[i] and jf > 0:
                    continue

                x_wa[i] = y_wa[i] + jf * ddx
                x_wa[i] = max(min(x_wa[i], u_wa[i]), l_wa[i])

                parameters.control.x = x_wa

                wrap_forward_run(
                    model.setup,
                    model.mesh,
                    model._input_data,
                    parameters,
                    model._output,
                    wrap_options,
                    wrap_returns,
                )
                nfg += 1

                if model._output.cost < gx:
                    z_wa = np.copy(x_wa)
                    gx = model._output.cost
                    ia = i
                    jfa = jf

        iaa = ia
        jfaa = jfa

        if ia > -1:
            y_wa = np.copy(z_wa)

            sdx *= clg
            sdx[ia] = (1.0 - clg) * jfa * ddx + clg * sdx[ia]

            iam += 1

            if iam + 1 > 2 * n:
                ddx *= 2
                iam = 0

            if gx < ga - 2:
                ga = gx
        else:
            ddx /= 2
            iam = -1

        if iter > 4 * n:
            for i in range(n):
                x_wa[i] = y_wa[i] + sdx[i]
                x_wa[i] = max(min(x_wa[i], u_wa[i]), l_wa[i])

            parameters.control.x = x_wa

            wrap_forward_run(
                model.setup,
                model.mesh,
                model._input_data,
                parameters,
                model._output,
                wrap_options,
                wrap_returns,
            )
            nfg += 1

            if model._output.cost < gx:
                gx = model._output.cost
                jfaa = 0
                y_wa = np.copy(x_wa)
                z_wa = np.copy(x_wa)

                if gx < ga - 2:
                    ga = gx

        ia = -1

        if iter % n == 0:
            if wrap_options.comm.verbose:
                print(
                    f"    At iterate{str(iter // n).rjust(7)}    nfg = {str(nfg).rjust(4)}"
                    f"    J = {gx:14.6f}    ddx = {ddx:5.2f}"
                )

            if "iter_cost" in return_options["keys"]:
                ret["iter_cost"] = np.append(ret["iter_cost"], gx)

        if ddx < 0.01:
            message = "CONVERGENCE: DDX < 0.01"
            break

    parameters.control.x = z_wa

    wrap_forward_run(
        model.setup,
        model.mesh,
        model._input_data,
        parameters,
        model._output,
        wrap_options,
        wrap_returns,
    )

    if "control_vector" in return_options["keys"]:
        ret["control_vector"] = parameters.control.x.copy()

    if "serr_mu" in return_options["keys"]:
        ret["serr_mu"] = model.get_serr_mu().copy()

    if "serr_sigma" in return_options["keys"]:
        ret["serr_sigma"] = model.get_serr_sigma().copy()

    if wrap_options.comm.verbose:
        print(f"    {message}")

    return ret


def _gradient_based_optimize_problem(
    x: np.ndarray,
    model: Model,
    parameters: ParametersDT,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
    callback: _OptimizeCallback,
) -> tuple[float, np.ndarray]:
    setattr(parameters.control, "x", x)

    parameters_b = parameters.copy()
    output_b = model._output.copy()
    output_b.cost = np.float32(1)

    wrap_forward_run_b(
        model.setup,
        model.mesh,
        model._input_data,
        parameters,
        parameters_b,
        model._output,
        output_b,
        wrap_options,
        wrap_returns,
    )

    if callback.iter_cost.size == 0:
        callback.iter_cost = np.append(callback.iter_cost, model._output.cost)

    callback.count_nfg += 1

    grad = parameters_b.control.x.copy()

    if callback.projg is None:
        callback.projg = _inf_norm(grad)

    else:
        callback.tmp_projg = _inf_norm(grad)

    return (model._output.cost, grad)
