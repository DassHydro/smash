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
from scipy.stats import truncnorm


__all__ = ["generate_samples", "SampleResult"]


class SampleResult(dict):
    """
    Represents the generated samples using `smash.generate_samples` method.

    Notes
    -----
    This class is essentially a subclass of dict with attribute accessors,
    which also have additional attributes not listed here depending on the specific names
    provided in the argument ``problem`` in the `smash.generate_samples` method.

    Attributes
    ----------
    generator: str
        The generator used to generate the samples.

    n_sample: int
        The number of generated samples.

    See Also
    --------
    smash.generate_samples: Generate a multiple set of spatially uniform Model parameters/states.

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

    generator : str, default 'uniform'
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
        In this case, truncated normal distribution could be used with respect to the boundary conditions defined by the above problem.

        .. note::
            If not given, the mean is the center of the parameter/state bound if in case of Gaussian generator.

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
    res : SampleResult
        The generated samples result represented as a ``SampleResult`` object.

    See Also
    --------
    smash.SampleResult: Represents the generated samples using `smash.generate_samples` method.
    Model.get_bound_constraints: Get the boundary constraints of the Model parameters/states.

    Examples
    --------
    Define the problem by a dictionary:

    >>> problem = {
    ...             'num_vars': 4,
    ...             'names': ['cp', 'cft', 'exc', 'lr'],
    ...             'bounds': [[1,2000], [1,1000], [-20,5], [1,1000]]
    ... }

    Generate samples with the uniform generator:

    >>> sr = smash.generate_samples(problem, n=3, random_state=99)
    >>> sr.keys()
    dict_keys(['cp', 'cft', 'exc', 'lr'])

    Access to generated sample result

    >>> sr.cp
    array([1344.8848387 ,  976.66872008, 1651.1648529 ])

    """

    ret_dict = {key: [] for key in problem["names"]}

    generator = generator.lower()

    ret_dict["generator"] = generator

    ret_dict["n_sample"] = n

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
            ret_dict[p] = np.append(ubi, np.random.uniform(low, upp, n - len(ubi)))

        elif generator in ["normal", "gaussian"]:
            if coef_std is None:
                sd = (upp - low) / 3

            else:
                sd = (upp - low) / coef_std

            if backg_sol is None:
                trunc_normal = _get_truncated_normal((low + upp) / 2, sd, low, upp)

            else:
                trunc_normal = _get_truncated_normal(ubi[0], sd, low, upp)

            ret_dict[p] = np.append(ubi, trunc_normal.rvs(size=n - len(ubi)))

        else:
            raise ValueError(
                f"Unknown generator '{generator}': Choices: {SAMPLE_GENERATORS}"
            )

    return SampleResult(ret_dict)


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
