from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy

from smash._constant import (
    ADAPTIVE_OPTIMIZER,
    GRADIENT_BASED_OPTIMIZER,
    OPTIMIZER_CLASS,
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
from smash.core.simulation.optimize._tools import (
    _get_lcurve_wjreg_best,
    _get_parameters_b,
    _handle_bayesian_optimize_control_prior,
    _inf_norm,
    _net2vect,
    _set_parameters_from_net,
)

# Used inside eval statement
from smash.factory.net._optimizers import SGD, Adagrad, Adam, RMSprop  # noqa: F401
from smash.fcore._mw_forward import forward_run as wrap_forward_run
from smash.fcore._mwd_options import OptionsDT
from smash.fcore._mwd_parameters_manipulation import (
    parameters_to_control as wrap_parameters_to_control,
)
from smash.fcore._mwd_returns import ReturnsDT

if TYPE_CHECKING:
    from typing import Any

    from smash.core.model.model import Model
    from smash.factory.net.net import Net
    from smash.fcore._mwd_parameters import ParametersDT


__all__ = [
    "BayesianOptimize",
    "Optimize",
    "bayesian_optimize",
    "optimize",
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

    internal_fluxes : dict[str, `numpy.ndarray`]
        A dictionary where keys are the names of the internal fluxes and the values are array of
        shape *(nrow, ncol, n)* representing an internal flux on the domain for each **time_step**.

    control_vector : `numpy.ndarray`
        An array of shape *(k,)* representing the control vector solution of the optimization (it can be
        transformed).

    net : `Net <factory.Net>`
        The trained neural network.

    cost : `float`
        Cost value.

    n_iter : `int`
        Number of iterations performed.

    projg : `numpy.ndarray`
        Projected gradient value (infinity norm of the Jacobian matrix).

    jobs : `float`
        Cost observation component value.

    jreg : `float`
        Cost regularization component value.

    lcurve_wjreg : `dict[str, Any]`
        A dictionary containing the wjreg lcurve data. The elements are:

        wjreg_opt : `float`
            The optimal wjreg value.

        wjreg_approx: `float`
            The approximative wjreg value evaluated with one optimization cycle only.

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

    internal_fluxes : dict[str, `numpy.ndarray`]
        A dictionary where keys are the names of the internal fluxes and the values are array of
        shape *(nrow, ncol, n)* representing an internal flux on the domain for each **time_step**.

    control_vector : `numpy.ndarray`
        An array of shape *(k,)* representing the control vector solution of the optimization (it can be
        transformed).

    cost : `float`
        Cost value.

    n_iter : `int`
        Number of iterations performed.

    projg : `numpy.ndarray`
        Projected gradient value (infinity norm of the Jacobian matrix).

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


# % The argument passed to the callback function is of type Optimize since
# BayesianOptimize and Optimize share the same structure and have common attribute names.
# Although the code only specifies Optimize, the documentation lists both types
# (Optimize and BayesianOptimize) to reflect their specific usage in different cases.


class _ScipyOptimizeCallback:
    # % Private callback class for external optimizer i.e. L-BFGS-B from scipy
    def __init__(self, callback: callable | None, verbose: bool):
        self.verbose = verbose

        self.iteration = 0

        self.nfg = 1
        self.count_nfg = 0

        self.cost = None

        self.projg = None
        self.projg_bak = None

        self.callback = callback

    def intermediate(self, intermediate_result: scipy.optimize.OptimizeResult):
        # % intermediate_result is required by callback function in scipy
        if self.verbose:
            msg = f"{' ' * 4}At iterate {self.iteration:>5}    nfg = {self.nfg:>5}    J = {self.cost:>.5e}"

            if self.projg is not None:
                msg += f"{' ' * 4}|proj g| = {self.projg:>.5e}"

            print(msg)

        self.iteration += 1

        self.nfg = self.count_nfg

        self.cost = intermediate_result.fun

        self.projg = self.projg_bak

        if self.callback is not None:
            self.callback(
                iopt=Optimize(
                    {
                        "control_vector": intermediate_result.x.copy(),
                        "cost": intermediate_result.fun,
                        "n_iter": self.iteration,
                        "projg": self.projg,
                    }
                )
            )

    def terminate(self, final_result: scipy.optimize.OptimizeResult):
        if self.verbose:
            msg = f"{' ' * 4}At iterate {self.iteration:>5}    nfg = {self.nfg:>5}    J = {self.cost:>.5e}"

            if self.projg is not None:
                msg += f"{' ' * 4}|proj g| = {self.projg:>.5e}"

            print(msg)

            print(f"{' ' * 4}{final_result.message}")


def _optimize_fast_wjreg(
    model: Model, options: OptionsDT, returns: ReturnsDT, optimize_options: dict, return_options: dict
) -> float:
    if options.comm.verbose:
        print(f"{' ' * 4}FAST WJREG CYCLE 1")

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
    _apply_optimizer(model, wparameters, options, returns, optimize_options, return_options, callback=None)

    jobs = returns.jobs
    jreg = returns.jreg

    wjreg = (jobs0 - jobs) / jreg

    return wjreg


def _optimize_lcurve_wjreg(
    model: Model, options: OptionsDT, returns: ReturnsDT, optimize_options: dict, return_options: dict
) -> tuple[float, dict]:
    if options.comm.verbose:
        print(f"{' ' * 4}L-CURVE WJREG CYCLE 1")

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
    _apply_optimizer(model, wparameters, options, returns, optimize_options, return_options, callback=None)

    cost = returns.cost
    jobs_min = returns.jobs
    jreg_min = 0.0
    jreg_max = returns.jreg
    wjreg_fast = 0.0

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
            print(f"{' ' * 4}L-CURVE WJREG CYCLE {i + 2}")

        wparameters = model._parameters.copy()
        _apply_optimizer(
            model, wparameters, options, returns, optimize_options, return_options, callback=None
        )

        cost_arr[i + 1] = returns.cost
        jobs_arr[i + 1] = returns.jobs
        jreg_arr[i + 1] = returns.jreg

    distance, wjreg = _get_lcurve_wjreg_best(cost_arr, jobs_arr, jreg_arr, wjreg_arr)

    lcurve = {
        "wjreg_opt": wjreg,
        "wjreg_approx": wjreg_fast,
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
    callback: callable | None = None,
) -> Model | tuple[Model, Optimize]:
    wmodel = model.copy()

    ret_optimize = wmodel.optimize(
        mapping,
        optimizer,
        optimize_options,
        cost_options,
        common_options,
        return_options,
        callback,
    )

    if ret_optimize is None:
        return wmodel
    else:
        return (wmodel, ret_optimize)


def _optimize(
    model: Model,
    mapping: str,
    optimizer: str,
    optimize_options: dict,
    cost_options: dict,
    common_options: dict,
    return_options: dict,
    callback: callable | None,
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

    if auto_wjreg == "fast":
        wrap_options.cost.wjreg = _optimize_fast_wjreg(
            model, wrap_options, wrap_returns, optimize_options, return_options
        )
        if wrap_options.comm.verbose:
            print(f"{' ' * 4}FAST WJREG LAST CYCLE. wjreg: {'{:.5e}'.format(wrap_options.cost.wjreg)}")
    elif auto_wjreg == "lcurve":
        wrap_options.cost.wjreg, lcurve_wjreg = _optimize_lcurve_wjreg(
            model, wrap_options, wrap_returns, optimize_options, return_options
        )
        if wrap_options.comm.verbose:
            print(f"{' ' * 4}L-CURVE WJREG LAST CYCLE. wjreg: {'{:.5e}'.format(wrap_options.cost.wjreg)}")
    else:
        pass

    pyret = _apply_optimizer(
        model, model._parameters, wrap_options, wrap_returns, optimize_options, return_options, callback
    )

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
        if any(k in SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS for k in ret):
            ret["time_step"] = return_options["time_step"].copy()
        return Optimize(ret)


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
    callback: callable | None = None,
) -> Model | tuple[Model, BayesianOptimize]:
    wmodel = model.copy()

    ret_bayesian_optimize = wmodel.bayesian_optimize(
        mapping,
        optimizer,
        optimize_options,
        cost_options,
        common_options,
        return_options,
        callback,
    )

    if ret_bayesian_optimize is None:
        return wmodel
    else:
        return (wmodel, ret_bayesian_optimize)


def _bayesian_optimize(
    model: Model,
    mapping: str,
    optimizer: str,
    optimize_options: dict,
    cost_options: dict,
    common_options: dict,
    return_options: dict,
    callback: callable | None,
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

    pyret = _apply_optimizer(
        model, model._parameters, wrap_options, wrap_returns, optimize_options, return_options, callback
    )

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
        if any(k in SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS for k in ret):
            ret["time_step"] = return_options["time_step"].copy()
        return BayesianOptimize(ret)


def _apply_optimizer(
    model: Model,
    parameters: ParametersDT,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
    optimize_options: dict,
    return_options: dict,
    callback: callable | None,
) -> dict:
    if wrap_options.optimize.optimizer == "sbs":
        ret = _sbs_optimize(
            model, parameters, wrap_options, wrap_returns, optimize_options, return_options, callback
        )

    elif wrap_options.optimize.optimizer in ["lbfgsb", "nelder-mead", "powell"]:  # scipy optimizers
        ret = _scipy_optimize(
            model, parameters, wrap_options, wrap_returns, optimize_options, return_options, callback
        )

    elif wrap_options.optimize.optimizer in ADAPTIVE_OPTIMIZER:
        if "net" in optimize_options:
            ret = _ann_adaptive_optimize(
                model, parameters, wrap_options, wrap_returns, optimize_options, return_options, callback
            )

        else:
            ret = _adaptive_optimize(
                model, parameters, wrap_options, wrap_returns, optimize_options, return_options, callback
            )

    # % Manually deallocate control
    parameters.control.dealloc()

    return ret


def _adaptive_optimize(
    model: Model,
    parameters: ParametersDT,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
    optimize_options: dict,
    return_options: dict,
    callback: callable | None,
) -> dict:
    ind = ADAPTIVE_OPTIMIZER.index(wrap_options.optimize.optimizer)
    func = eval(OPTIMIZER_CLASS[ind])

    adap_opt = func(learning_rate=optimize_options["learning_rate"])

    maxiter = optimize_options["termination_crit"]["maxiter"]
    early_stopping = optimize_options["termination_crit"]["early_stopping"]

    wrap_parameters_to_control(
        model.setup,
        model.mesh,
        model._input_data,
        parameters,
        wrap_options,
    )

    x = parameters.control.x.copy()

    l_control = parameters.control.l.copy()
    u_control = parameters.control.u.copy()

    has_lower_bound = np.isin(parameters.control.nbd, [1, 2])
    has_upper_bound = np.isin(parameters.control.nbd, [2, 3])

    # % First evaluation
    parameters_b = _get_parameters_b(model, parameters, wrap_options, wrap_returns)
    grad = parameters_b.control.x.copy()
    projg = _inf_norm(grad)

    opt_info = {"cost": np.inf}  # only used for early_stopping

    if wrap_options.comm.verbose:
        print(
            f"{' ' * 4}At iterate {0:>5}    nfg = {1:>5}    "
            f"J = {model._output.cost:>.5e}    |proj g| = {projg:>.5e}"
        )

    for ite in range(1, maxiter + 1):
        # % Gradient-based parameter update
        x = adap_opt.update(x, grad)

        # % Check bounds condition
        x = np.where(has_lower_bound, np.maximum(x, l_control), x)
        x = np.where(has_upper_bound, np.minimum(x, u_control), x)

        # % Set control values and run adjoint model to get new gradients
        setattr(parameters.control, "x", x)

        parameters_b = _get_parameters_b(model, parameters, wrap_options, wrap_returns)
        grad = parameters_b.control.x.copy()

        projg = _inf_norm(grad)

        # % Stop if early stopping is set and met
        if early_stopping:
            if model._output.cost < opt_info["cost"] or ite == 1:
                opt_info["ite"] = ite
                opt_info["control"] = np.copy(x)
                opt_info["cost"] = model._output.cost
                opt_info["projg"] = projg

            elif (
                ite - opt_info["ite"] > early_stopping
            ):  # stop training if the loss values do not decrease through early_stopping consecutive
                # iterations
                if wrap_options.comm.verbose:
                    print(
                        f"{' ' * 4}EARLY STOPPING: NO IMPROVEMENT for {early_stopping} CONSECUTIVE ITERATIONS"
                    )
                break

        if callback is not None:
            callback(
                iopt=Optimize(
                    {"control_vector": np.copy(x), "cost": model._output.cost, "n_iter": ite, "projg": projg}
                )
            )

        if wrap_options.comm.verbose:
            print(
                f"{' ' * 4}At iterate {ite:>5}    nfg = {ite + 1:>5}    "
                f"J = {model._output.cost:>.5e}    |proj g| = {projg:>.5e}"
            )

            if ite == maxiter:
                print(f"{' ' * 4}STOP: TOTAL NO. of ITERATIONS REACHED LIMIT")

    if early_stopping:
        if opt_info["ite"] < maxiter:
            if wrap_options.comm.verbose:
                print(
                    f"{' ' * 4}Revert to iteration {opt_info['ite']} with "
                    f"J = {opt_info['cost']:.5e} due to early stopping"
                )

            x = opt_info["control"]
            projg = opt_info["projg"]

    # % Apply final control and forward run for updating final states
    setattr(parameters.control, "x", x)

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
        ret["control_vector"] = x

    if "n_iter" in return_options["keys"]:
        ret["n_iter"] = opt_info.get("ite", maxiter)

    if "projg" in return_options["keys"]:
        ret["projg"] = projg

    if "serr_mu" in return_options["keys"]:
        ret["serr_mu"] = model.get_serr_mu().copy()

    if "serr_sigma" in return_options["keys"]:
        ret["serr_sigma"] = model.get_serr_sigma().copy()

    return ret


def _ann_adaptive_optimize(
    model: Model,
    parameters: ParametersDT,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
    optimize_options: dict,
    return_options: dict,
    callback: callable | None,
) -> Net:
    # % Preprocessing input descriptors and normalization
    l_desc = model._input_data.physio_data.l_descriptor
    u_desc = model._input_data.physio_data.u_descriptor

    desc = model._input_data.physio_data.descriptor.copy()
    desc = (desc - l_desc) / (u_desc - l_desc)  # normalize input descriptors

    # % Train regionalization network
    net = optimize_options["net"]

    if net.layers[0].layer_name() == "Dense":
        desc = desc.reshape(-1, desc.shape[-1])

    istop = net._fit_d2p(
        desc,
        model,
        parameters,
        wrap_options,
        wrap_returns,
        wrap_options.optimize.optimizer,
        optimize_options["parameters"],
        optimize_options["learning_rate"],
        optimize_options["random_state"],
        optimize_options["termination_crit"]["maxiter"],
        optimize_options["termination_crit"]["early_stopping"],
        wrap_options.comm.verbose,
        callback,
    )

    # % Revert model parameters if early stopped (nn_parameters have been reverted inside net._fit_d2p)
    if istop:
        _set_parameters_from_net(net, desc, optimize_options["parameters"], parameters)

    # % Reset control with ann mapping (do not apply to rr_parameters and rr_initial_states)
    wrap_parameters_to_control(
        model.setup,
        model.mesh,
        model._input_data,
        parameters,
        wrap_options,
    )

    # % Forward run for updating final states
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

    if "net" in return_options["keys"]:
        ret["net"] = net

    if "control_vector" in return_options["keys"]:
        ret["control_vector"] = np.append(parameters.control.x, _net2vect(net))

    if "n_iter" in return_options["keys"]:
        ret["n_iter"] = istop if istop else optimize_options["termination_crit"]["maxiter"]

    if "projg" in return_options["keys"]:
        ret["projg"] = net.history["proj_grad"][istop - 1]

    return ret


def _scipy_optimize(
    model: Model,
    parameters: ParametersDT,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
    optimize_options: dict,
    return_options: dict,
    callback: callable | None,
) -> dict:
    wrap_parameters_to_control(
        model.setup,
        model.mesh,
        model._input_data,
        parameters,
        wrap_options,
    )

    x0 = parameters.control.x.copy()

    # % Set None values for unbounded/semi-unbounded controls to pass to scipy optimize functions
    # which requires None to identify unbounded values
    l_control = np.where(np.isin(parameters.control.nbd, [0, 3]), None, parameters.control.l)
    u_control = np.where(np.isin(parameters.control.nbd, [0, 1]), None, parameters.control.u)

    scipy_callback = _ScipyOptimizeCallback(callback, wrap_options.comm.verbose)

    if wrap_options.optimize.optimizer == "lbfgsb":
        res_optimize = scipy.optimize.minimize(
            _gradient_based_optimize_problem,
            x0,
            args=(model, parameters, wrap_options, wrap_returns, scipy_callback),
            method="L-BFGS-B",
            jac=True,
            bounds=tuple(zip(l_control, u_control)),
            callback=scipy_callback.intermediate,
            options={
                "maxiter": optimize_options["termination_crit"]["maxiter"],
                "ftol": 2.22e-16 * optimize_options["termination_crit"]["factr"],
                "gtol": optimize_options["termination_crit"]["pgtol"],
            },
        )

    elif wrap_options.optimize.optimizer == "nelder-mead":
        res_optimize = scipy.optimize.minimize(
            _gradient_free_optimize_problem,
            x0,
            args=(model, parameters, wrap_options, wrap_returns, scipy_callback),
            method="Nelder-Mead",
            bounds=tuple(zip(l_control, u_control)),
            callback=scipy_callback.intermediate,
            options={
                "maxiter": optimize_options["termination_crit"]["maxiter"],
                "xatol": optimize_options["termination_crit"]["xatol"],
                "fatol": optimize_options["termination_crit"]["fatol"],
            },
        )

    elif wrap_options.optimize.optimizer == "powell":
        res_optimize = scipy.optimize.minimize(
            _gradient_free_optimize_problem,
            x0,
            args=(model, parameters, wrap_options, wrap_returns, scipy_callback),
            method="Powell",
            bounds=tuple(zip(l_control, u_control)),
            callback=scipy_callback.intermediate,
            options={
                "maxiter": optimize_options["termination_crit"]["maxiter"],
            },
        )

    scipy_callback.terminate(res_optimize)

    # % Apply final control and forward run for updating final states
    setattr(parameters.control, "x", res_optimize.x)

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
        ret["control_vector"] = res_optimize.x

    if "n_iter" in return_options["keys"]:
        ret["n_iter"] = res_optimize.nit

    if (wrap_options.optimize.optimizer in GRADIENT_BASED_OPTIMIZER) and ("projg" in return_options["keys"]):
        ret["projg"] = scipy_callback.projg

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
    optimize_options: dict,
    return_options: dict,
    callback: callable | None,
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

    message = "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"

    if wrap_options.comm.verbose:
        print(f"{' ' * 4}At iterate {0:>5}    nfg = {nfg:>5}    J = {gx:>.5e}    ddx = {ddx:>4.2f}")

    for iter in range(1, optimize_options["termination_crit"]["maxiter"] * n + 1):
        dxn = min(dxn, ddx)
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

        converged = ddx < 0.01

        if (iter % n == 0) or converged:
            iteration = iter // n + (iter % n > 0)

            if callback is not None:
                callback(iopt=Optimize({"control_vector": np.copy(z_wa), "cost": gx, "n_iter": iteration}))

            if wrap_options.comm.verbose:
                print(
                    f"{' ' * 4}At iterate {iteration:>5}    nfg = {nfg:>5}    J = {gx:>.5e}    "
                    f"ddx = {ddx:>4.2f}"
                )

        if converged:
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
        ret["control_vector"] = z_wa

    if "n_iter" in return_options["keys"]:
        ret["n_iter"] = iteration

    if "serr_mu" in return_options["keys"]:
        ret["serr_mu"] = model.get_serr_mu().copy()

    if "serr_sigma" in return_options["keys"]:
        ret["serr_sigma"] = model.get_serr_sigma().copy()

    if wrap_options.comm.verbose:
        print(f"{' ' * 4}{message}")

    return ret


def _gradient_based_optimize_problem(
    x: np.ndarray,
    model: Model,
    parameters: ParametersDT,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
    scipy_callback: _ScipyOptimizeCallback,
) -> tuple[float, np.ndarray]:
    # % Set control values
    setattr(parameters.control, "x", x)

    # % Get gradient J wrt control vector
    parameters_b = _get_parameters_b(model, parameters, wrap_options, wrap_returns)
    grad = parameters_b.control.x.copy()

    # % Callback
    scipy_callback.count_nfg += 1

    if scipy_callback.cost is None:
        scipy_callback.cost = model._output.cost

    if scipy_callback.projg is None:
        scipy_callback.projg = _inf_norm(grad)

    else:
        scipy_callback.projg_bak = _inf_norm(grad)

    return (model._output.cost, grad)


def _gradient_free_optimize_problem(
    x: np.ndarray,
    model: Model,
    parameters: ParametersDT,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
    scipy_callback: _ScipyOptimizeCallback,
) -> float:
    # % Set control values
    setattr(parameters.control, "x", x)

    # % Forward run to get cost
    wrap_forward_run(
        model.setup,
        model.mesh,
        model._input_data,
        parameters,
        model._output,
        wrap_options,
        wrap_returns,
    )

    # % Callback
    scipy_callback.count_nfg += 1

    if scipy_callback.cost is None:
        scipy_callback.cost = model._output.cost

    return model._output.cost
