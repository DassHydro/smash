from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm

from smash.core.simulation.run.run import _forward_run

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.core.simulation.run.run import ForwardRun
    from smash.factory.samples.samples import Samples
    from smash.util._typing import AnyTuple


def _compute_density(
    samples: Samples,
    optimized_parameters: dict,
    active_cell: np.ndarray,
) -> dict:
    density = {}

    for p, optim_param in optimized_parameters.items():
        dst = getattr(samples, "_dst_" + p)
        density[p] = np.tile(
            dst, (*active_cell.shape, 1)
        )  # spatialized density (*active_cell.shape, n_sample)

        if isinstance(optim_param, np.ndarray):
            estimated_cell = np.zeros(active_cell.shape)

            for ac in [0, 1]:  # Iterate on two blocs active/inactive cell
                mask = np.where(active_cell == ac)

                if np.all(
                    [
                        np.allclose(
                            optim_param[..., i][mask],
                            optim_param[..., i][mask][0],
                        )
                        for i in range(optim_param.shape[-1])
                    ]
                ):  # if optim_param[mask] contain only uniform values
                    unf_optim_param = optim_param[mask][0, :]

                    if np.allclose(unf_optim_param, unf_optim_param[0]):
                        estimted_density = np.ones(unf_optim_param.shape)
                    else:
                        estimted_density = gaussian_kde(unf_optim_param)(unf_optim_param)

                    density[p][mask] *= estimted_density  # compute joint-probability

                    estimated_cell[mask] = True

            for i, j in np.ndindex(active_cell.shape):  # Iterate on all grid cells
                if not estimated_cell[i, j]:
                    unf_optim_param_ij = optim_param[i, j, :]

                    if np.allclose(unf_optim_param_ij, unf_optim_param_ij[0]):
                        estimted_density = np.ones(unf_optim_param_ij.shape)
                    else:
                        estimted_density = gaussian_kde(unf_optim_param_ij)(unf_optim_param_ij)

                    density[p][i, j] *= estimted_density  # compute joint-probability

    return density


def _estimate_parameter(
    prior_data: np.ndarray,
    cost_values: np.ndarray,
    density: np.ndarray,
    alpha: float,
) -> AnyTuple:
    # prior_data: 3D-array
    # cost_values: 1D-array
    # density: 3D-array

    likelihood = np.exp(-(2**alpha) * (cost_values / min(cost_values) - 1) ** 2)

    weighting = likelihood * density  # 3D-array

    sum_weighting = np.sum(weighting, axis=2)  # 2D-array

    estim_param = 1 / sum_weighting * np.sum(prior_data * weighting, axis=2)  # 2D-array

    var_param = (
        1 / sum_weighting * np.sum((prior_data - estim_param[..., np.newaxis]) ** 2 * weighting, axis=2)
    )  # 2D-array
    var_param = np.mean(var_param)

    mean_prior_data = np.mean(prior_data, axis=2)  # 2D-array

    mahal_distance = np.mean(np.square(estim_param - mean_prior_data)) / var_param

    return (estim_param, mahal_distance)


def _forward_run_with_estimated_parameters(
    alpha: float,
    model: Model,
    prior_data: dict,
    density: dict,
    cost: np.ndarray,
    cost_options: dict,
    common_options: dict,
    return_options: dict,
) -> (ForwardRun | None, dict):
    mahal_distance = 0

    for p in prior_data.keys():
        param_p, distance_p = _estimate_parameter(prior_data[p], cost, density[p], alpha)

        if p in model.rr_parameters.keys:
            model.set_rr_parameters(p, param_p)

        elif p in model.rr_initial_states.keys:
            model.set_rr_initial_states(p, param_p)

        # % In case we have other kind of parameters. Should be unreachable.
        else:
            pass

        mahal_distance += distance_p

    # Remove forward run verbose
    common_options["verbose"] = False

    ret_forward_run = _forward_run(
        model,
        cost_options=cost_options,
        common_options=common_options,
        return_options=return_options,
    )

    return ret_forward_run, dict(
        zip(
            ["mahal_dist", "cost"],
            [mahal_distance / len(prior_data), model._output.cost],
        )
    )


def _lcurve_forward_run_with_estimated_parameters(
    alpha: np.ndarray,
    *args_forward_run_with_estimated_parameters: AnyTuple,
) -> (ForwardRun | None, dict):
    l_cost = np.zeros(alpha.size)
    l_mahal_distance = np.zeros(alpha.size)

    for i in tqdm(range(alpha.size), desc="    L-curve Computing"):
        _, lcurve_i = _forward_run_with_estimated_parameters(
            alpha[i], *args_forward_run_with_estimated_parameters
        )

        l_mahal_distance[i] = lcurve_i["mahal_dist"]
        l_cost[i] = lcurve_i["cost"]

    l_mahal_distance_scaled = (l_mahal_distance - np.min(l_mahal_distance)) / (
        np.max(l_mahal_distance) - np.min(l_mahal_distance)
    )
    l_cost_scaled = (l_cost - np.min(l_cost)) / (np.max(l_cost) - np.min(l_cost))

    regul_term = np.square(l_mahal_distance_scaled) + np.square(l_cost_scaled)

    alpha_opt = alpha[np.argmin(regul_term)]

    ret_forward_run, _ = _forward_run_with_estimated_parameters(
        alpha_opt, *args_forward_run_with_estimated_parameters
    )

    return ret_forward_run, dict(
        zip(
            ["mahal_dist", "cost", "alpha", "alpha_opt"],
            [l_mahal_distance, l_cost, alpha, alpha_opt],
        )
    )
