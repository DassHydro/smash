from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver._mwd_setup import SetupDT

from smash.core._constant import (
    STRUCTURE_PARAMETERS,
    STRUCTURE_STATES,
    SAMPLE_GENERATORS,
    PROBLEM_KEYS,
)

import warnings

import numpy as np
import pandas as pd
from scipy.stats import truncnorm


__all__ = ["generate_samples"]


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

        - 'num_vars' : the number of Model parameters/states.
        - 'names' : the name of Model parameters/states.
        - 'bounds' : the upper and lower bounds of each Model parameters/states (a sequence of (min, max)).

        .. hint::
            This problem can be created using the Model object. See `smash.Model.get_bound_constraints` for more.

    generator : str, default uniform
        Samples generator. Should be one of

        - 'uniform'
        - 'normal' or 'gaussian'

    n : int, default 1000
        Number of generated samples.

    random_state : int or None, default None
        Random seed used to generate samples.

        .. note::
            If not given, generates parameters sets with a random seed.

    backg_sol : numpy.ndarray or None, default None
        Spatially uniform prior parameters/states could be included in generated sets, and are
        used as the mean when generating with Gaussian distribution.

        .. note::
            If not given, the mean is the center of the parameter/state bound if in case of Gaussian generator, otherwise,
            there is no background solution included in generated sets.

    coef_std : float or None
        A coefficient related to the standard deviation in case of Gaussian generator:

        .. math::
                std = \\frac{u - l}{coef\\_std}

        where :math:`u` and :math:`l` are the upper and lower bounds of Model parameters/states.

        .. note::
            If not given, a default value for this coefficient will be assigned to define the standard deviation:

        .. math::
                std = \\frac{u - l}{3}

    Returns
    -------
    res : pandas.DataFrame
        A dataframe with generated samples.

    See Also
    --------
    Model.get_bound_constraints: Get the boundary constraints of the Model parameters/states.

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
                f"Unknown generator '{generator}': Choices: {SAMPLE_GENERATORS}"
            )

    return df


def _get_truncated_normal(mean: float, sd: float, low: float, upp: float):

    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def _get_bound_constraints(setup: SetupDT, states: bool):

    if states:
        control_vector = STRUCTURE_STATES[setup.structure]

    else:
        control_vector = STRUCTURE_PARAMETERS[setup.structure]

    bounds = []

    for name in control_vector:

        if name in setup._states_name:

            ind = np.argwhere(setup._states_name == name)

            l = setup._optimize.lb_states[ind].item()
            u = setup._optimize.ub_states[ind].item()

        else:

            ind = np.argwhere(setup._parameters_name == name)

            l = setup._optimize.lb_parameters[ind].item()
            u = setup._optimize.ub_parameters[ind].item()

        bounds += [[l, u]]

    problem = {
        "num_vars": len(control_vector),
        "names": control_vector,
        "bounds": bounds,
    }

    return problem


def _standardize_problem(problem: dict | None, setup: SetupDT, states: bool):

    if problem is None:

        problem = _get_bound_constraints(setup, states)

    elif isinstance(problem, dict):

        prl_keys = problem.keys()

        if not all(k in prl_keys for k in PROBLEM_KEYS):

            raise KeyError(
                f"Problem dictionary should be defined with required keys {PROBLEM_KEYS}"
            )

        unk_keys = tuple(k for k in prl_keys if k not in PROBLEM_KEYS)

        if unk_keys:

            warnings.warn(f"Unknown key(s) found in the problem definition {unk_keys}")

    else:
        raise TypeError("The problem definition must be a dictionary or None")

    return problem
