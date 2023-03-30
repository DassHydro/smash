from __future__ import annotations

from smash.core._constant import (
    STRUCTURE_PARAMETERS,
    STRUCTURE_STATES,
    SAMPLE_GENERATORS,
    PROBLEM_KEYS,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver._mwd_setup import SetupDT

import warnings

import numpy as np
import pandas as pd
from scipy.stats import truncnorm


__all__ = ["generate_samples", "SampleResult"]


class SampleResult(dict):
    """
    Represents the generated samples using `smash.generate_samples` method.

    Notes
    -----
    This class is essentially a subclass of dict with attribute accessors and two additional methods, which are
    `SampleResult.to_numpy` and `SampleResult.to_dataframe`.
    This also have additional attributes not listed here depending on the specific names
    provided in the argument ``problem`` in the `smash.generate_samples` method.

    Attributes
    ----------
    generator : str
        The generator used to generate the samples.

    n_sample : int
        The number of generated samples.

    See Also
    --------
    smash.generate_samples: Generate a multiple set of spatially uniform Model parameters/states.

    Examples
    --------
    >>> problem = {"num_vars": 2, "names": ["cp", "lr"], "bounds": [[1,200], [1,500]]}
    >>> sr = smash.generate_samples(problem, n=5, random_state=1)

    Convert the result to a numpy.ndarray:

    >>> sr.to_numpy(axis=-1)
    array([[ 83.98737894,  47.07695879],
           [144.34457419,  93.94384548],
           [  1.02276059, 173.43480279],
           [ 61.16418195, 198.98696964],
           [ 30.20442227, 269.86955027]])

    Convert the result to a pandas.DataFrame:

    >>> sr.to_dataframe()
               cp          lr
    0   83.987379   47.076959
    1  144.344574   93.943845
    2    1.022761  173.434803
    3   61.164182  198.986970
    4   30.204422  269.869550

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

    def to_numpy(self, axis=0):
        """
        Convert the `SampleResult` object to a numpy.ndarray.

        The attribute arrays are stacked along a user-specified axis of the resulting array in alphabetical order
        based on the names of the Model parameters/states.

        Parameters
        ----------
        axis : int, default 0
            The axis along which the generated samples of each Model parameter/state will be joined.

        Returns
        -------
        res : numpy.ndarray
            The `SampleResult` object as a numpy.ndarray.

        """
        keys = sorted(self._problem["names"])

        return np.stack([self[k] for k in keys], axis=axis)

    def to_dataframe(self):
        """
        Convert the `SampleResult` object to a pandas.DataFrame.

        Returns
        -------
        res : pandas.DataFrame
            The SampleResult object as a pandas.DataFrame.
        """

        return pd.DataFrame({k: self[k] for k in self._problem["names"]})


def generate_samples(
    problem: dict,
    generator: str = "uniform",
    n: int = 1000,
    random_state: int | None = None,
    mean: np.ndarray | None = None,
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
        - 'bounds' : the upper and lower bounds of each Model parameter/state (a sequence of ``(min, max)``).

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

    mean : dict or None, default None
        If the samples are generated using a Gaussian distribution, ``mean`` is used to define the mean of the distribution for each Model parameter/state.
        It is a dictionary where keys are the name of the parameters/states defined in the ``problem`` argument.
        In this case, the truncated normal distribution may be used with respect to the boundary conditions defined in the ``problem`` argument.
        None value inside the dictionary will be filled in with the center of the parameter/state bounds.

        .. note::
            If not given, the mean of the distribution will be set to the center of the parameter/state bounds if a Gaussian distribution is used.

    coef_std : float or None
        A coefficient related to the standard deviation in case of Gaussian generator:

        .. math::
                std = \\frac{u - l}{coef\\_std}

        where :math:`u` and :math:`l` are the upper and lower bounds of Model parameters/states.

        .. note::
            If not given, a default value for this coefficient will be assigned to define the standard deviation if a Gaussian distribution is used:

            .. math::
                std = \\frac{u - l}{3}

    Returns
    -------
    res : SampleResult
        The generated samples result represented as a `SampleResult` object.

    See Also
    --------
    SampleResult: Represents the generated samples using `smash.generate_samples` method.
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
    >>> sr.cp
    array([1344.8848387 ,  976.66872008, 1651.1648529 ])

    """

    generator, mean = _standardize_generate_samples_args(problem, generator, mean)

    ret_dict = {key: [] for key in problem["names"]}

    ret_dict["generator"] = generator

    ret_dict["n_sample"] = n

    ret_dict["_problem"] = problem.copy()

    if random_state is not None:
        np.random.seed(random_state)

    for i, p in enumerate(problem["names"]):
        low = problem["bounds"][i][0]
        upp = problem["bounds"][i][1]

        if generator == "uniform":
            ret_dict[p] = np.random.uniform(low, upp, n)

            ret_dict["_" + p] = 1 / n * np.ones(n)

        elif generator in ["normal", "gaussian"]:
            if coef_std is None:
                sd = (upp - low) / 3

            else:
                sd = (upp - low) / coef_std

            trunc_normal = _get_truncated_normal(mean[p], sd, low, upp)

            ret_dict[p] = trunc_normal.rvs(size=n)

            ret_dict["_" + p] = trunc_normal.pdf(ret_dict[p])

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


def _standardize_generate_samples_args(problem: dict, generator: str, user_mean: dict):
    if isinstance(problem, dict):  # simple check problem
        _standardize_problem(problem, None, None)

    else:
        raise TypeError("problem must be a dictionary")

    if isinstance(generator, str):  # check generator
        generator = generator.lower()

        if generator not in SAMPLE_GENERATORS:
            raise ValueError(
                f"Unknown generator '{generator}': Choices: {SAMPLE_GENERATORS}"
            )

        elif generator in ["normal", "gaussian"]:
            # check mean
            mean = dict(zip(problem["names"], np.mean(problem["bounds"], axis=1)))

            if user_mean is None:
                pass

            elif isinstance(user_mean, dict):
                for name, um in user_mean.items():
                    if not name in problem["names"]:
                        warnings.warn(
                            f"Key {name} does not match any existing names in the problem definition {problem['names']}"
                        )

                    if isinstance(um, (int, float)):
                        mean.update({name: um})

                    else:
                        raise TypeError("mean value must be float or integer")

            else:
                raise TypeError("mean must be None or a dictionary")

        else:
            mean = user_mean

    else:
        raise TypeError("generator must be a string")

    return generator, mean
