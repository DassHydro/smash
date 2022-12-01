from __future__ import annotations

from smash.core._constant import STRUCTURE_PARAMETERS

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver._mwd_setup import SetupDT

import numpy as np
import pandas as pd
from SALib.sample import saltelli
from scipy.stats import truncnorm


__all__ = ["generate_samples"]


def _get_truncated_normal(mean: float, sd: float, low: float, upp: float):

    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def generate_samples(
    problem: dict,
    generator: str = "uniform",
    n: int = 1000,
    random_state: int | None = None,
    backg_sol: np.ndarray | None = None,
    coef_std: float | None = None,
):
    """
    Generate a multiple set of spatially uniform Model parameters/states.

    Parameters
    ----------
    problem : dict
        Problem definition. The keys are

        - 'num_vars': The number of Model parameters/states.
        - 'names': The name of Model parameters/states
        - 'bounds': The upper and lower bounds of each Model parameters/states (a sequence of (min, max))

    generator : str, default uniform
        Samples generator. Should be one of

        - 'uniform'
        - 'normal' or 'gaussian'
        - 'saltelli'

    n : int, default 1000
        Number of generated samples.
        In case of Saltelli generator, this is the number of trajectories to generate for each model parameter (ideally a power of 2).
        Then the number of sample to generate for all model parameters is equal to :math:`N(2D+2)`
        where :math:`D` is the number of model parameters.

        See `here <https://salib.readthedocs.io/en/latest/api.html>`__ for more details.

    random_state : int or None, default None
        Random seed used to generate sample, except Saltelli, which is determinist generator
        and do not require a random seed.

        .. note::
            If not given, generates parameters sets with a random seed with Gaussian or uniform generators.

    backg_sol : numpy.ndarray or None, default None
        Prior solutions could be included in parameters sets, except Saltelli generator, and are
        used as the mean when generating with Gaussian distribution.

        .. note::
            If not given, the mean is the center of the parameter bound if in case of Gaussian generator, otherwise,
            there is no background solution included in the generated parameter sets.

    coef_std : float or None
        A coefficient related to the standard deviation in case of Gaussian generator:

        .. math::
                std = \\frac{u - l}{coef\_std}

        .. note::
            If not given, a default value for this coefficient will be assigned to define the standard deviation:

        .. math::
                std = \\frac{u - l}{3}

    Returns
    -------
    res : pandas.DataFrame
        res with all generated samples

    Examples
    --------
    Define the problem by a dictionary:

    >>> problem = {
    ...             'num_vars': 4,
    ...             'names': ['cp', 'cft', 'exc', 'lr'],
    ...             'bounds': [[1,2000], [1,1000], [-20,5], [1,1000]]
    ... }

    Generate samples with `uniform` generator:

    >>> smash.generate_samples(problem, n=5, random_state=99)
                    cp         cft        exc          lr
        0  1344.884839  566.051802  -0.755174  396.058590
        1   976.668720  298.324876  -1.330822  973.982341
        2  1651.164853   47.649025 -10.564027  524.890301
        3    63.861329  990.636772  -7.646314   94.519480
        4  1616.291877    7.818907   3.223710  813.495104

    """

    df = pd.DataFrame(columns=problem["names"])

    generator = generator.lower()

    if generator == "saltelli":  # determinist generator

        sample = saltelli.sample(problem, n)

        df[df.keys()] = sample

    else:  # non-determinist generator
        if random_state is not None:
            np.random.seed(random_state)

        for i, p in enumerate(problem["names"]):

            low = problem["bounds"][i][0]
            upp = problem["bounds"][i][1]

            if backg_sol is None:
                ubi = []

            else:
                ubi = [backg_sol[i]]

            if generator == "uniform":

                df[p] = np.append(ubi, np.random.uniform(low, upp, n - len(ubi)))

            elif generator in ["normal", "gaussian"]:

                if coef_std is None:
                    sd = (upp - low) / 3

                else:
                    sd = (upp - low) / coef_std

                if backg_sol is None:
                    trunc_normal = _get_truncated_normal((low + upp) / 2, sd, low, upp)

                else:
                    trunc_normal = _get_truncated_normal(ubi[0], sd, low, upp)

                df[p] = np.append(ubi, trunc_normal.rvs(size=n - len(ubi)))

            else:
                raise ValueError(
                    f"Unknown generator '{generator}': Choices: ['uniform', 'normal', 'gaussian', 'saltelli']"
                )

    return df


def _get_generate_samples_problem(setup: SetupDT):

    bounds = []

    for name in STRUCTURE_PARAMETERS[setup.structure]:

        if name in setup._parameters_name:

            ind = np.argwhere(setup._parameters_name == name)

            l = setup._optimize.lb_parameters[ind].item()
            u = setup._optimize.ub_parameters[ind].item()

            bounds += [[l, u]]

    problem = {
        "num_vars": len(STRUCTURE_PARAMETERS[setup.structure]),
        "names": STRUCTURE_PARAMETERS[setup.structure],
        "bounds": bounds,
    }

    return problem
