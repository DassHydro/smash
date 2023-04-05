from __future__ import annotations

from smash.solver._mw_forward import forward

from smash.core._constant import OPTIM_FUNC
from smash.core._event_segmentation import _mask_event

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model
    from smash.core.generate_samples import SampleResult

import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import multiprocessing as mp
from tqdm import tqdm

__all__ = ["BayesResult"]


class BayesResult(dict):
    """
    Represents the Bayesian estimation or optimization result.

    Notes
    -----
    This class is essentially a subclass of dict with attribute accessors.

    Attributes
    ----------
    data : dict
        Rrepresenting the generated spatially uniform Model parameters/sates and the corresponding cost values after
        running the simulations on this dataset. The keys are 'cost' and the names of Model parameters/states considered.

    density : dict
        Representing the estimated distribution at pixel scale of the Model parameters/states after
        running the simulations. The keys are the names of the Model parameters/states.

    lcurve : dict
        The optimization results on the regularization parameter if the L-curve approach is used. The keys are

        - 'alpha' : a list of regularization parameters to be optimized.
        - 'cost' : a list of corresponding cost values.
        - 'mahal_dist' : a list of corresponding Mahalanobis distance values.
        - 'var' : a list of corresponding dictionaries. The keys are the names of the Model parameters/states, and each represents its variance.
        - 'alpha_opt' : the optimal value of the regularization parameter.

    See Also
    --------
    Model.bayes_estimate: Estimate prior Model parameters/states using Bayesian approach.
    Model.bayes_optimize: Optimize the Model using Bayesian approach.

    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return "\n".join(
                [k.rjust(m) + ": " + repr(v) for k, v in sorted(self.items())]
            )
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def _bayes_computation(
    instance: Model,
    sample: SampleResult,
    alpha: int | float | list,
    bw_method: str | None,
    weights: np.ndarray | None,
    algorithm: str | None,
    mapping: str | None,
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
    event_seg: dict,
    bounds: np.ndarray | None,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    verbose: bool,
    options: dict | None,
    ncpu: int,
) -> BayesResult:
    # % Prior solution
    prior_data = {}

    # % returns
    ret_data = {}
    ret_density = {}
    ret_lcurve = {}

    # % verbose
    if verbose:
        _bayes_message(sample, alpha)

    # % Build data from sample
    res_simu = _multi_simu(
        instance,
        sample,
        algorithm,
        mapping,
        jobs_fun,
        wjobs_fun,
        event_seg,
        bounds,
        wgauge,
        ost,
        options,
        ncpu,
    )

    prior_data["cost"] = np.array(res_simu["cost"])

    ret_data["cost"] = np.array(res_simu["cost"])

    for p in sample._problem["names"]:
        ret_data[p] = getattr(sample, p)

        dat_p = np.dstack(res_simu[p])

        prior_data[p] = dat_p

        ret_density[p] = np.ones(dat_p.shape)

    # % Density compute
    active_mask = np.where(instance.mesh.active_cell == 1)

    _compute_density(
        sample,
        prior_data,
        active_mask,
        ret_density,
        algorithm,
        bw_method,
        weights,
    )

    # % Bayes compute
    if isinstance(alpha, list):
        _lcurve_compute_param(
            instance,
            sample,
            jobs_fun,
            wjobs_fun,
            event_seg,
            wgauge,
            ost,
            active_mask,
            prior_data,
            ret_density,
            ret_lcurve,
            alpha,
        )

    else:
        _compute_param(
            instance,
            sample,
            jobs_fun,
            wjobs_fun,
            event_seg,
            wgauge,
            ost,
            active_mask,
            prior_data,
            ret_density,
            alpha,
        )

    return BayesResult(
        dict(zip(["data", "density", "lcurve"], [ret_data, ret_density, ret_lcurve]))
    )


def _bayes_message(sr: SampleResult, alpha: int | float | list):
    sp4 = " " * 4

    lcurve = True if isinstance(alpha, list) else False

    ret = []

    ret.append(f"{sp4}Parameters/States set size: {sr.n_sample}")
    ret.append(f"Sample generator: {sr.generator}")

    ret.append(f"L-curve approach: {lcurve}")

    print(f"\n{sp4}".join(ret) + "\n")


### BUILD sample ###


def _run(
    instance: Model,
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
    event_seg: dict,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
):
    ### SETTING MODEL TO COMPUTE COST VALUES ###

    # % send mask_event to Fortran in case of event signatures based optimization
    if any([fn[0] == "E" for fn in jobs_fun]):
        instance.setup._optimize.mask_event = _mask_event(instance, **event_seg)

    # % Set values for Fortran derived type variables
    instance.setup._optimize.jobs_fun = jobs_fun
    instance.setup._optimize.wjobs_fun = wjobs_fun
    instance.setup._optimize.wgauge = wgauge

    st = pd.Timestamp(instance.setup.start_time)
    instance.setup._optimize.optimize_start_step = (
        ost - st
    ).total_seconds() / instance.setup.dt + 1

    ###

    # % FORWARD MODEL

    cost = np.float32(0)

    forward(
        instance.setup,
        instance.mesh,
        instance.input_data,
        instance.parameters,
        instance.parameters.copy(),
        instance.states,
        instance.states.copy(),
        instance.output,
        cost,
    )


def _unit_simu(
    i: int,
    instance: Model,
    sample: SampleResult,
    algorithm: str | None,
    mapping: str | None,
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
    event_seg: dict | None,
    bounds: np.ndarray,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    options: dict | None,
) -> dict:
    # % SET PARAMS/STATES
    for name in sample._problem["names"]:
        if name in instance.setup._parameters_name:
            setattr(instance.parameters, name, getattr(sample, name)[i])

        else:
            setattr(instance.states, name, getattr(sample, name)[i])

    # % SIMU (RUN OR OPTIMIZE)
    if algorithm is None:
        _run(instance, jobs_fun, wjobs_fun, event_seg, wgauge, ost)

    else:
        OPTIM_FUNC[algorithm](
            instance,
            sample._problem["names"],
            mapping,
            jobs_fun,
            wjobs_fun,
            event_seg,
            bounds,
            wgauge,
            ost,
            False,
            **options,
        )

    res = {}

    res["cost"] = instance.output.cost

    for name in sample._problem["names"]:
        if name in instance.setup._parameters_name:
            res[name] = np.copy(getattr(instance.parameters, name))

        else:
            res[name] = np.copy(getattr(instance.states, name))

    return res


def _multi_simu(
    instance: Model,
    sample: SampleResult,
    algorithm: str | None,
    mapping: str | None,
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
    event_seg: dict,
    bounds: np.ndarray,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    options: dict | None,
    ncpu: int,
) -> dict:
    if algorithm is None:
        pgbar_mess = "</> Running forward Model on multiset"

    else:
        pgbar_mess = "</> Optimizing Model parameters on multiset"

    if ncpu > 1:
        list_instance = [instance.copy() for i in range(sample.n_sample)]

        pool = mp.Pool(ncpu)
        list_result = pool.starmap(
            _unit_simu,
            [
                (
                    i,
                    instance,
                    sample,
                    algorithm,
                    mapping,
                    jobs_fun,
                    wjobs_fun,
                    event_seg,
                    bounds,
                    wgauge,
                    ost,
                    options,
                )
                for i, instance in tqdm(enumerate(list_instance), desc=pgbar_mess)
            ],
        )
        pool.close()

    elif ncpu == 1:
        list_result = []

        for i in tqdm(range(sample.n_sample), desc=pgbar_mess):
            list_result.append(
                _unit_simu(
                    i,
                    instance,
                    sample,
                    algorithm,
                    mapping,
                    jobs_fun,
                    wjobs_fun,
                    event_seg,
                    bounds,
                    wgauge,
                    ost,
                    options,
                )
            )

    else:
        raise ValueError(f"ncpu should be a positive integer, not {ncpu}")

    res_keys = list(sample._problem["names"])
    res_keys.append("cost")
    res = {k: [] for k in res_keys}

    for result in list_result:
        for k in res.keys():
            res[k].append(result[k])

    return res


### DENSITY ESTIMATE ###


def _compute_density(
    sample: SampleResult,
    data: dict,
    active_mask: np.ndarray,
    density: dict,
    algorithm: str | None,
    bw_method: str | None,
    weights: np.ndarray | None,
):
    coord = np.dstack([active_mask[0], active_mask[1]])[0]

    for p in sample._problem["names"]:
        dat_p = np.copy(data[p])

        if algorithm == "l-bfgs-b":  # variational Bayes optim (HD)
            for c in coord:
                density[p][c[0], c[1]] = gaussian_kde(
                    dat_p[c[0], c[1]], bw_method=bw_method, weights=weights
                )(dat_p[c[0], c[1]])

        else:
            if isinstance(algorithm, str):
                u_dis = np.mean(dat_p[active_mask], axis=0)

                uniform_density = gaussian_kde(
                    u_dis, bw_method=bw_method, weights=weights
                )(
                    u_dis
                )  # global Bayes optim (LD)

            else:
                uniform_density = getattr(sample, "_" + p)  # Bayes estim (LD)

            for c in coord:
                density[p][c[0], c[1]] = uniform_density


### BAYES ESTIMATE AND L-CURVE


def _compute_mean_U(
    U: np.ndarray, J: np.ndarray, rho: np.ndarray, alpha: float, mask: np.ndarray
) -> tuple:
    # U is 3-D array
    # rho is 3-D array
    # J is 1-D array

    L = np.exp(-(2**alpha) * (J / min(J) - 1) ** 2)  # likelihood
    Lrho = L * rho  # 3-D array

    C = np.sum(Lrho, axis=2)  # C is 2-D array

    U_alp = 1 / C * np.sum(U * Lrho, axis=2)  # 2-D array

    varU = 1 / C * np.sum((U - U_alp[..., np.newaxis]) ** 2 * Lrho, axis=2)  # 2-D array
    varU = np.mean(varU[mask])

    Uinf = np.mean(U, axis=2)  # 2-D array

    D_alp = np.mean(np.square(U_alp - Uinf)[mask]) / varU

    return U_alp, varU, D_alp


def _compute_param(
    instance: Model,
    sample: SampleResult,
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
    event_seg: dict,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    active_mask: np.ndarray,
    prior_data: dict,
    ret_density: dict,
    alpha: int | float,
) -> tuple:
    D_alp = []

    J = np.copy(prior_data["cost"])

    var = {}

    for name in sample._problem["names"]:
        U = np.copy(prior_data[name])
        rho = np.copy(ret_density[name])

        u, v, d = _compute_mean_U(U, J, rho, alpha, active_mask)

        if name in instance.setup._parameters_name:
            setattr(instance.parameters, name, u)

        else:
            setattr(instance.states, name, u)

        D_alp.append(d)

        var[name] = v

    _run(
        instance,
        jobs_fun,
        wjobs_fun,
        event_seg,
        wgauge,
        ost,
    )

    return instance.output.cost, np.mean(D_alp), var


def _lcurve_compute_param(
    instance: Model,
    sample: SampleResult,
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
    event_seg: dict,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    active_mask: np.ndarray,
    prior_data: dict,
    ret_density: dict,
    ret_lcurve: dict,
    alpha: list,
):
    cost = []
    D_alp = []
    var = []

    for alpha_i in alpha:
        co, d_alp, vr = _compute_param(
            instance,
            sample,
            jobs_fun,
            wjobs_fun,
            event_seg,
            wgauge,
            ost,
            active_mask,
            prior_data,
            ret_density,
            alpha_i,
        )

        cost.append(co)
        D_alp.append(d_alp)
        var.append(vr)

    cost_scaled = (cost - np.min(cost)) / (np.max(cost) - np.min(cost))
    D_alp_scaled = (D_alp - np.min(D_alp)) / (np.max(D_alp) - np.min(D_alp))

    d = np.square(cost_scaled) + np.square(D_alp_scaled)

    alpha_opt = alpha[np.argmin(d)]

    _compute_param(
        instance,
        sample,
        jobs_fun,
        wjobs_fun,
        event_seg,
        wgauge,
        ost,
        active_mask,
        prior_data,
        ret_density,
        alpha_opt,
    )

    ret_lcurve["alpha"] = alpha

    ret_lcurve["alpha_opt"] = alpha_opt

    ret_lcurve["mahal_dist"] = D_alp

    ret_lcurve["cost"] = cost

    ret_lcurve["var"] = var
