from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from smash._constant import (
    ADAPTIVE_OPTIMIZER,
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
    _forward_run_b,
    _get_lcurve_wjreg_best,
    _handle_bayesian_optimize_control_prior,
    _inf_norm,
    _net_to_parameters,
    _net_to_vect,
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
                f"{' '*4}At iterate{str(self.iterations).rjust(7)}    nfg = {str(self.nfg).rjust(4)}"
                f"{' '*4}J = {self.iter_cost[-1]:14.6f}    |proj g| = {self.projg:14.6f}"
            )

        self.iterations += 1

        self.nfg = self.count_nfg

        self.iter_cost = np.append(self.iter_cost, intermediate_result.fun)

        self.iter_projg = np.append(self.iter_projg, self.projg)
        self.projg = self.projg_bak

    def termination(self, final_result: scipy_OptimizeResult):
        if self.verbose:
            print(
                f"{' '*4}At iterate{str(self.iterations).rjust(7)}    nfg = {str(self.nfg).rjust(4)}"
                f"{' '*4}J = {final_result.fun:14.6f}    |proj g| = {self.projg:14.6f}"
            )
            print(f"{' '*4}{final_result.message}")

        self.iter_projg = np.append(self.iter_projg, self.projg)


def _optimize_fast_wjreg(
    model: Model, options: OptionsDT, returns: ReturnsDT, optimize_options: dict, return_options: dict
) -> float:
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
    _apply_optimizer(model, wparameters, options, returns, optimize_options, return_options)

    jobs = returns.jobs
    jreg = returns.jreg

    wjreg = (jobs0 - jobs) / jreg

    return wjreg


def _optimize_lcurve_wjreg(
    model: Model, options: OptionsDT, returns: ReturnsDT, optimize_options: dict, return_options: dict
) -> (float, dict):
    if options.comm.verbose:
        print(f"{' '*4}L-CURVE WJREG CYCLE 1")

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
    _apply_optimizer(model, wparameters, options, returns, optimize_options, return_options)

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
            print(f"{' '*4}L-CURVE WJREG CYCLE {i + 2}")

        wparameters = model._parameters.copy()
        _apply_optimizer(model, wparameters, options, returns, optimize_options, return_options)

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

    if auto_wjreg == "fast":
        wrap_options.cost.wjreg = _optimize_fast_wjreg(
            model, wrap_options, wrap_returns, optimize_options, return_options
        )
        if wrap_options.comm.verbose:
            print(f"{' '*4}FAST WJREG LAST CYCLE. wjreg: {'{:.6f}'.format(wrap_options.cost.wjreg)}")
    elif auto_wjreg == "lcurve":
        wrap_options.cost.wjreg, lcurve_wjreg = _optimize_lcurve_wjreg(
            model, wrap_options, wrap_returns, optimize_options, return_options
        )
        if wrap_options.comm.verbose:
            print(f"{' '*4}L-CURVE WJREG LAST CYCLE. wjreg: {'{:.6f}'.format(wrap_options.cost.wjreg)}")
    else:
        pass

    pyret = _apply_optimizer(
        model, model._parameters, wrap_options, wrap_returns, optimize_options, return_options
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
        if any(k in SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS for k in ret.keys()):
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

    pyret = _apply_optimizer(
        model, model._parameters, wrap_options, wrap_returns, optimize_options, return_options
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
        if any(k in SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS for k in ret.keys()):
            ret["time_step"] = return_options["time_step"].copy()
        return BayesianOptimize(ret)


def _apply_optimizer(
    model: Model,
    parameters: ParametersDT,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
    optimize_options: dict,
    return_options: dict,
) -> dict:
    if wrap_options.optimize.optimizer == "sbs":
        ret = _sbs_optimize(model, parameters, wrap_options, wrap_returns, return_options)

    elif wrap_options.optimize.optimizer == "lbfgsb":
        ret = _lbfgsb_optimize(model, parameters, wrap_options, wrap_returns, return_options)

    elif wrap_options.optimize.optimizer in ADAPTIVE_OPTIMIZER:
        ret = _adaptive_optimize(
            model, parameters, wrap_options, wrap_returns, optimize_options, return_options
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
) -> dict:
    ret = {}

    if "net" in optimize_options.keys():
        net = _reg_ann_adaptive_optimize(model, parameters, wrap_options, wrap_returns, optimize_options)

        if "net" in return_options["keys"]:
            ret["net"] = net

        if "iter_cost" in return_options["keys"]:
            ret["iter_cost"] = net.history["loss_train"]

        if "iter_projg" in return_options["keys"]:
            ret["iter_projg"] = net.history["proj_grad"]

        if "control_vector" in return_options["keys"]:
            ret["control_vector"] = np.append(parameters.control.x, _net_to_vect(net))

    else:
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

        has_lower_bound = l_control != -99
        has_upper_bound = u_control != -99

        # % First evaluation
        parameters_b = _forward_run_b(model, parameters, wrap_options, wrap_returns)
        grad = parameters_b.control.x.copy()
        projg = _inf_norm(grad)

        opt_info = {"ite": 0, "value": model._output.cost, "control": x}

        if "iter_cost" in return_options["keys"]:
            ret["iter_cost"] = np.array([model._output.cost])

        if "iter_projg" in return_options["keys"]:
            ret["iter_projg"] = np.array([projg])

        if wrap_options.comm.verbose:
            print(
                f"{' '*4}At iterate{str(0).rjust(7)}    nfg = {str(1).rjust(4)}"
                f"{' '*4}J = {model._output.cost:14.6f}    |proj g| = {projg:14.6f}"
            )

        for ite in range(1, maxiter + 1):
            # % Gradient-based parameter update
            x = adap_opt.update(x, grad)

            # % Check bounds condition
            x = np.where(has_lower_bound, np.maximum(x, l_control), x)
            x = np.where(has_upper_bound, np.minimum(x, u_control), x)

            # % Run adjoint to get new gradients
            grad = _jac_optimize_problem(x, model, parameters, wrap_options, wrap_returns)
            projg = _inf_norm(grad)

            if "iter_cost" in return_options["keys"]:
                ret["iter_cost"] = np.append(ret["iter_cost"], model._output.cost)

            if "iter_projg" in return_options["keys"]:
                ret["iter_projg"] = np.append(ret["iter_projg"], projg)

            if early_stopping:
                if model._output.cost < opt_info["value"]:
                    opt_info["ite"] = ite
                    opt_info["value"] = model._output.cost
                    opt_info["control"] = x.copy()

                else:
                    if (
                        ite - opt_info["ite"] > early_stopping
                    ):  # stop training if the loss values do not decrease through early_stopping consecutive
                        # iterations
                        if wrap_options.comm.verbose:
                            print(
                                f"{' '*4}EARLY STOPPING: NO IMPROVEMENT for {early_stopping} CONSECUTIVE "
                                f"ITERATIONS"
                            )
                        break

            if wrap_options.comm.verbose:
                print(
                    f"{' '*4}At iterate{str(ite).rjust(7)}    nfg = {str(ite+1).rjust(4)}"
                    f"{' '*4}J = {model._output.cost:14.6f}    |proj g| = {projg:14.6f}"
                )

                if ite == maxiter:
                    print(f"{' '*4}STOP: TOTAL NO. of ITERATIONS REACHED LIMIT")

        if early_stopping:
            if opt_info["ite"] < maxiter:
                if wrap_options.comm.verbose:
                    print(
                        f"{' '*4}Reverting to iteration {opt_info['ite']} with "
                        f"J = {opt_info['value']:.6f} due to early stopping"
                    )

                x = opt_info["control"]

        # % Run forward model to update final states and control vector (if early stopped)
        _optimize_problem(x, model, parameters, wrap_options, wrap_returns)

        if "control_vector" in return_options["keys"]:
            ret["control_vector"] = parameters.control.x.copy()

        if "serr_mu" in return_options["keys"]:
            ret["serr_mu"] = model.get_serr_mu().copy()

        if "serr_sigma" in return_options["keys"]:
            ret["serr_sigma"] = model.get_serr_sigma().copy()

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

    l_wa = parameters.control.l.copy()
    u_wa = parameters.control.u.copy()

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
    jfaa = 0
    nfg = 1

    ret = {}

    message = "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"

    if wrap_options.comm.verbose:
        print(
            f"{' '*4}At iterate{str(0).rjust(7)}    nfg = {str(nfg).rjust(4)}"
            f"{' '*4}J = {gx:14.6f}    ddx = {ddx:5.2f}"
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
                parameters.control.l = l_wa
                parameters.control.u = u_wa

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
            parameters.control.l = l_wa
            parameters.control.u = u_wa

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
                    f"{' '*4}At iterate{str(iter // n).rjust(7)}    nfg = {str(nfg).rjust(4)}"
                    f"{' '*4}J = {gx:14.6f}    ddx = {ddx:5.2f}"
                )

            if "iter_cost" in return_options["keys"]:
                ret["iter_cost"] = np.append(ret["iter_cost"], gx)

        if ddx < 0.01:
            message = "CONVERGENCE: DDX < 0.01"
            break

    parameters.control.x = z_wa
    parameters.control.l = l_wa
    parameters.control.u = u_wa

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
        print(f"{' '*4}{message}")

    return ret


def _reg_ann_adaptive_optimize(
    model: Model,
    parameters: ParametersDT,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
    optimize_options: dict,
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

    # % Change the mapping to trigger distributed control to get distributed gradients
    wrap_options.optimize.mapping = "distributed"

    early_stopped = net._fit_d2p(
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
    )

    # % Reset mapping to ann once fit_d2p done
    wrap_options.optimize.mapping = "ann"

    # % Revert model parameters if early stopped (nn_parameters have been reverted inside net._fit_d2p)
    if early_stopped:
        _net_to_parameters(net, desc, optimize_options["parameters"], parameters, model.mesh.flwdir.shape)

    # % Reset control with ann mapping
    wrap_parameters_to_control(
        model.setup,
        model.mesh,
        model._input_data,
        parameters,
        wrap_options,
    )  # only apply to nn_parameters if used since mapping has been reverted to ann

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

    return net


def _reg_ann_adaptive_optimize(
    model: Model,
    parameters: ParametersDT,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
    optimize_options: dict,
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

    # % Change the mapping to trigger distributed control to get distributed gradients
    wrap_options.optimize.mapping = "distributed"

    early_stopped = net._fit_d2p(
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
    )

    # % Reset mapping to ann once fit_d2p done
    wrap_options.optimize.mapping = "ann"

    # % Revert model parameters if early stopped (nn_parameters have been reverted inside net._fit_d2p)
    if early_stopped:
        _net_to_parameters(net, desc, optimize_options["parameters"], parameters, model.mesh.flwdir.shape)

    # % Reset control with ann mapping
    wrap_parameters_to_control(
        model.setup,
        model.mesh,
        model._input_data,
        parameters,
        wrap_options,
    )  # only apply to nn_parameters if used since mapping has been reverted to ann

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

    return net


def _gradient_based_optimize_problem(
    x: np.ndarray,
    model: Model,
    parameters: ParametersDT,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
    callback: _OptimizeCallback,
) -> tuple[float, np.ndarray]:
    setattr(parameters.control, "x", x)

    parameters_b = _forward_run_b(model, parameters, wrap_options, wrap_returns)

    if callback.iter_cost.size == 0:
        callback.iter_cost = np.append(callback.iter_cost, model._output.cost)

    callback.count_nfg += 1

    grad = parameters_b.control.x.copy()

    if callback is not None:
        if callback.projg is None:
            callback.projg = _inf_norm(grad)

        else:
            callback.projg_bak = _inf_norm(grad)

    return (model._output.cost, grad)
