from __future__ import annotations

from smash.solver._mw_forward import forward

from smash.core._constant import OPTIM_FUNC
from smash.core._event_segmentation import _mask_event
from smash.core.generate_samples import generate_samples

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import multiprocessing as mp


class BayesResult(dict):
    """
    Represents the Bayesian estimation or optimization results.

    Notes
    -----
    This class is essentially a subclass of dict with attribute accessors.

    Attributes
    ----------
    data : dict
        Rrepresenting the generated Model parameters/sates and the corresponding cost values after
        running the simulations on this dataset. The keys are 'cost' and the names of Model parameters/states considered.
    density : dict
        Representing the estimated distribution at pixel scale of Model parameters/states after
        running the simulations. The keys are the names of Model parameters/states considered.
    l_curve : dict
        The optimization results on the regularisation parameter if the L-curve approach is used. The keys are

        - 'k' : a list of regularisation parameters to optimize.
        - 'cost' : a list of corresponding cost values.
        - 'Mahalanobis_distance' : a list of corresponding Mahalanobis distance values.
        - 'var' : a list of corresponding dictionaries. Each represents the variance of Model parameters/states. The keys are the names of Model parameters/states considered.
        - 'k_opt' : the optimal regularisation value.

    See Also
    --------
    Model.Bayes_estimate: Estimate prior Model parameters/states using Bayesian approach.
    Model.Bayes_optimize: Optimize the Model using Bayesian approach.

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
    generator: str,
    n: int,
    random_state: int | None,
    backg_sol: np.ndarray | None,
    coef_std: float | None,
    k: int | float | list,
    density_estimate: bool | None,
    bw_method: str | None,
    weights: np.ndarray | None,
    algorithm: str | None,
    control_vector: np.ndarray,
    mapping: str | None,
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
    bounds: np.ndarray,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    verbose: bool,
    options: dict | None,
    ncpu: int,
) -> BayesResult:

    #% returns
    ret_data = {}
    ret_density = {}
    ret_l_curve = {}

    #% verbose
    if verbose:
        _bayes_message(n, generator, backg_sol, density_estimate, k)

    #% standardize density_estimate
    if density_estimate is None:

        if generator.lower() == "uniform":
            density_estimate = False

        else:
            density_estimate = True

    ### Generate sample
    problem = {
        "num_vars": len(control_vector),
        "names": list(control_vector),
        "bounds": [list(bound) for bound in bounds],
    }
    sample = generate_samples(
        problem=problem,
        generator=generator,
        n=n,
        random_state=random_state,
        backg_sol=backg_sol,
        coef_std=coef_std,
    )

    ### Build data from sample

    res_simu = _multi_simu(
        instance,
        sample,
        algorithm,
        control_vector,
        mapping,
        jobs_fun,
        wjobs_fun,
        bounds,
        wgauge,
        ost,
        verbose,
        options,
        ncpu,
    )

    ret_data["cost"] = np.array(res_simu["cost"])

    for p in control_vector:

        dat_p = np.dstack(res_simu[p])

        ret_data[p] = dat_p

        ret_density[p] = np.ones(dat_p.shape)

    ### Density estimate

    active_mask = np.where(instance.mesh.active_cell == 1)

    if density_estimate:
        _estimate_density(
            ret_data,
            active_mask,
            ret_density,
            algorithm,
            control_vector,
            bw_method,
            weights,
        )

    ### Bayes compute

    if isinstance(k, list):

        _lcurve_compute_param(
            instance,
            jobs_fun,
            wjobs_fun,
            wgauge,
            ost,
            active_mask,
            control_vector,
            ret_data,
            ret_density,
            ret_l_curve,
            k,
        )

    else:

        _compute_param(
            instance,
            jobs_fun,
            wjobs_fun,
            wgauge,
            ost,
            active_mask,
            control_vector,
            ret_data,
            ret_density,
            k,
        )

    return BayesResult(
        dict(zip(["data", "density", "l_curve"], [ret_data, ret_density, ret_l_curve]))
    )


def _bayes_message(
    n_set: int,
    generator: str,
    backg_sol: np.ndarray | None,
    density_estimate: bool,
    k: int | float | list,
):

    sp4 = " " * 4

    if isinstance(k, list):
        lcurve = True

    else:
        lcurve = False

    ret = []

    ret.append(f"{sp4}Parameters/States set size: {n_set}")
    ret.append(f"Sample generator: {generator}")
    ret.append(f"Spatially uniform prior parameters/states: {backg_sol}")

    if density_estimate is not None:
        ret.append(f"Density estimation: {density_estimate}")

    ret.append(f"L-curve approach: {lcurve}")

    print(f"\n{sp4}".join(ret) + "\n")


### BUILD sample ###


def _run(
    instance: Model,
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
):

    ### SETTING MODEL TO COMPUTE COST VALUES ###

    #% send mask_event to Fortran in case of event signatures based optimization
    if any([fn[0] == "E" for fn in jobs_fun]):
        instance.setup._optimize.mask_event = _mask_event(instance)

    #% Set values for Fortran derived type variables
    instance.setup._optimize.jobs_fun = jobs_fun
    instance.setup._optimize.wjobs_fun = wjobs_fun
    instance.setup._optimize.wgauge = wgauge

    st = pd.Timestamp(instance.setup.start_time)
    instance.setup._optimize.optimize_start_step = (
        ost - st
    ).total_seconds() / instance.setup.dt + 1

    ###

    #% FORWARD MODEL

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
    sample: pd.DataFrame,
    algorithm: str | None,
    control_vector: np.ndarray,
    mapping: str | None,
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
    bounds: np.ndarray,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    verbose: bool,
    options: dict | None,
) -> dict:

    #% SET PARAMS/STATES
    for name in control_vector:

        if name in instance.setup._parameters_name:
            setattr(instance.parameters, name, sample.iloc[i][name])

        else:
            setattr(instance.states, name, sample.iloc[i][name])

    #% verbose
    if verbose:
        print(f"....SET {i+1} computing....")

    #% SIMU (RUN OR OPTIMIZE)
    if algorithm is None:
        _run(instance, jobs_fun, wjobs_fun, wgauge, ost)

    else:
        OPTIM_FUNC[algorithm](
            instance,
            control_vector,
            mapping,
            jobs_fun,
            wjobs_fun,
            bounds,
            wgauge,
            ost,
            verbose,
            **options,
        )

    res = {}

    res["cost"] = instance.output.cost

    for name in control_vector:

        if name in instance.setup._parameters_name:
            res[name] = np.copy(getattr(instance.parameters, name))

        else:
            res[name] = np.copy(getattr(instance.states, name))

    return res


def _multi_simu(
    instance: Model,
    sample: pd.DataFrame,
    algorithm: str | None,
    control_vector: np.ndarray,
    mapping: str | None,
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
    bounds: np.ndarray,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    verbose: bool,
    options: dict | None,
    ncpu: int,
) -> dict:

    if ncpu > 1:

        list_instance = [instance.copy() for i in range(len(sample))]

        pool = mp.Pool(ncpu)
        list_result = pool.starmap(
            _unit_simu,
            [
                (
                    i,
                    instance,
                    sample,
                    algorithm,
                    control_vector,
                    mapping,
                    jobs_fun,
                    wjobs_fun,
                    bounds,
                    wgauge,
                    ost,
                    verbose,
                    options,
                )
                for i, instance in enumerate(list_instance)
            ],
        )
        pool.close()

    elif ncpu == 1:

        list_result = []

        for i in range(len(sample)):

            list_result.append(
                _unit_simu(
                    i,
                    instance,
                    sample,
                    algorithm,
                    control_vector,
                    mapping,
                    jobs_fun,
                    wjobs_fun,
                    bounds,
                    wgauge,
                    ost,
                    verbose,
                    options,
                )
            )

    else:
        raise ValueError(f"ncpu should be a positive integer, not {ncpu}")

    res_keys = list(control_vector)
    res_keys.append("cost")
    res = {k: [] for k in res_keys}

    for result in list_result:

        for k in res.keys():

            res[k].append(result[k])

    return res


### DENSITY ESTIMATE ###


def _estimate_density(
    data: dict,
    active_mask: np.ndarray,
    density: dict,
    algorithm: str | None,
    control_vector: np.ndarray,
    bw_method: str | None,
    weights: np.ndarray | None,
):

    coord = np.dstack([active_mask[0], active_mask[1]])[0]

    for p in control_vector:

        dat_p = np.copy(data[p])

        if algorithm == "l-bfgs-b":

            for c in coord:
                density[p][c[0], c[1]] = gaussian_kde(
                    dat_p[c[0], c[1]], bw_method=bw_method, weights=weights
                )(dat_p[c[0], c[1]])

        else:

            u_dis = np.mean(dat_p[active_mask], axis=0)
            uniform_density = gaussian_kde(u_dis, bw_method=bw_method, weights=weights)(
                u_dis
            )

            for c in coord:
                density[p][c[0], c[1]] = uniform_density


### BAYES ESTIMATE AND L-CURVE


def _compute_mean_U(
    U: np.ndarray, J: np.ndarray, rho: np.ndarray, k: float, mask: np.ndarray
) -> tuple:

    # U is 3-D array
    # rho is 3-D array
    # J is 1-D array

    L = np.exp(-(2**k) * (J / min(J) - 1) ** 2)  # likelihood
    Lrho = L * rho  # 3-D array

    C = np.sum(Lrho, axis=2)  # C is 2-D array

    U_k = 1 / C * np.sum(U * Lrho, axis=2)  # 2-D array

    varU = 1 / C * np.sum((U - U_k[..., np.newaxis]) ** 2 * Lrho, axis=2)  # 2-D array
    varU = np.mean(varU[mask])

    Uinf = np.mean(U, axis=2)  # 2-D array

    Dk = np.mean(np.square(U_k - Uinf)[mask]) / varU

    return U_k, varU, Dk


def _compute_param(
    instance: Model,
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    active_mask: np.ndarray,
    control_vector: np.ndarray,
    ret_data: dict,
    ret_density: dict,
    k: int | float,
) -> tuple:

    Dk = []

    J = np.copy(ret_data["cost"])

    var = {}

    for name in control_vector:

        U = np.copy(ret_data[name])
        rho = np.copy(ret_density[name])

        u, v, d = _compute_mean_U(U, J, rho, k, active_mask)

        if name in instance.setup._parameters_name:
            setattr(instance.parameters, name, u)

        else:
            setattr(instance.states, name, u)

        Dk.append(d)

        var[name] = v

    _run(
        instance,
        jobs_fun,
        wjobs_fun,
        wgauge,
        ost,
    )

    return instance.output.cost, np.mean(Dk), var


def _lcurve_compute_param(
    instance: Model,
    jobs_fun: np.ndarray,
    wjobs_fun: np.ndarray,
    wgauge: np.ndarray,
    ost: pd.Timestamp,
    active_mask: np.ndarray,
    control_vector: np.ndarray,
    ret_data: dict,
    ret_density: dict,
    ret_l_curve: dict,
    k: list,
):

    cost = []
    Dk = []
    var = []

    for k_i in k:

        co, dk, vr = _compute_param(
            instance,
            jobs_fun,
            wjobs_fun,
            wgauge,
            ost,
            active_mask,
            control_vector,
            ret_data,
            ret_density,
            k_i,
        )

        cost.append(co)
        Dk.append(dk)
        var.append(vr)

    cost_scaled = (cost - np.min(cost)) / (np.max(cost) - np.min(cost))
    Dk_scaled = (Dk - np.min(Dk)) / (np.max(Dk) - np.min(Dk))

    d = np.square(cost_scaled) + np.square(Dk_scaled)

    kopt = k[np.argmin(d)]

    _compute_param(
        instance,
        jobs_fun,
        wjobs_fun,
        wgauge,
        ost,
        active_mask,
        control_vector,
        ret_data,
        ret_density,
        kopt,
    )

    ret_l_curve["k"] = k

    ret_l_curve["k_opt"] = kopt

    ret_l_curve["Mahalanobis_distance"] = Dk

    ret_l_curve["cost"] = cost

    ret_l_curve["var"] = var
