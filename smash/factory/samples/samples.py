from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from smash.factory.samples._standardize import _standardize_generate_samples_args

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from smash.util._typing import Numeric


__all__ = ["Samples", "generate_samples"]


class Samples:
    """
    Represents the generated samples result.

    See Also
    --------
    smash.factory.generate_samples: Generate a multiple set of variables.

    Notes
    -----
    This class may have additional attributes not listed here depending on the specific names provided
    in the argument ``problem`` in the `smash.factory.generate_samples <factory.generate_samples>` method.

    Attributes
    ----------
    generator : `str`
        The generator used to generate the samples.

    n_sample : `int`
        The number of generated samples.
    """

    def __init__(self, data: dict | None = None):
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

    def slice(self, end: int, start: int = 0) -> Samples:
        """
        Slice the Samples object.

        The attribute arrays are sliced along a user-specified start and end index.

        Parameters
        ----------
        end : `int`
            The end index of the slice.

        start : `int`, default 0
            The start index of the slice. Must be lower than **end**.

        Returns
        -------
        slice : `Samples`
            The Samples object sliced according to **start** and **end** arguments.

        Examples
        --------
        >>> from smash.factory import generate_samples
        >>> problem = {"num_vars": 2, "names": ["cp", "llr"], "bounds": [[1, 200], [1, 500]]}

        Generate samples

        >>> sample = generate_samples(problem, n=5, random_state=1)
        >>> sample.to_dataframe()
                   cp         llr
        0   83.987379   47.076959
        1  144.344574   93.943845
        2    1.022761  173.434803
        3   61.164182  198.986970
        4   30.204422  269.869550

        Slice the first two sets

        >>> sample_slice = sample.slice(2)
        >>> sample_slice.to_dataframe()
                   cp        llr
        0   83.987379  47.076959
        1  144.344574  93.943845

        Slice between start and end indexes

        >>> sample_slice = sample.slice(start=3, end=5)
        >>> sample_slice.to_dataframe()
                  cp        llr
        0  61.164182  198.98697
        1  30.204422  269.86955
        """

        if end < start:
            raise ValueError(f"start argument {start} must be lower than end argument {end}")

        if start < 0:
            raise ValueError(f"start argument {start} must be greater or equal to 0")

        if end > self.n_sample:
            raise ValueError(f"end argument {end} must be lower or equal to the sample size {self.n_sample}")

        slc_n = end - start

        slc_names = list(self._problem["names"]) + ["_dst_" + key for key in self._problem["names"]]

        slc_dict = {key: getattr(self, key)[start:end] for key in slc_names}

        slc_dict["generator"] = self.generator

        slc_dict["n_sample"] = slc_n

        slc_dict["_problem"] = self._problem.copy()

        return Samples(slc_dict)

    def iterslice(self, by: int = 1) -> Iterator[Samples]:
        """
        Iterate on the Samples object by slices.

        Parameters
        ----------
        by : `int`, default 1
            The size of the sample slice.
            If **by** is not a multiple of the sample size :math:`n` the last slice iteration size will
            be updated to the maximum range. It results in :math:`k=\\lfloor{\\frac{n}{by}}\\rfloor`
            iterations of size :math:`by` and one last iteration of size :math:`n - k \\times by`.

        Yields
        ------
        slice : `Samples`
            The Samples object sliced according to **by** arguments.

        See Also
        --------
        Samples.slice: Slice the Samples object.

        Examples
        --------
        >>> from smash.factory import generate_samples
        >>> problem = {"num_vars": 2, "names": ["cp", "llr"], "bounds": [[1, 200], [1, 500]]}

        Generate samples

        >>> sample = generate_samples(problem, n=5, random_state=1)
        >>> sample.to_dataframe()
                   cp         llr
        0   83.987379   47.076959
        1  144.344574   93.943845
        2    1.022761  173.434803
        3   61.164182  198.986970
        4   30.204422  269.869550

        Iterate on each set

        >>> for sample_slice in sample.iterslice():
                sample_slice.to_numpy(axis=-1)
        array([[83.98737894, 47.07695879]])
        array([[144.34457419,  93.94384548]])
        array([[  1.02276059, 173.43480279]])
        array([[ 61.16418195, 198.98696964]])
        array([[ 30.20442227, 269.86955027]])

        Iterate on pairs of sets

        >>> for sample_slice in sample.iterslice(2):
                sample_slice.to_numpy(axis=-1)
        array([[ 83.98737894,  47.07695879],
               [144.34457419,  93.94384548]])
        array([[  1.02276059, 173.43480279],
               [ 61.16418195, 198.98696964]])
        array([[ 30.20442227, 269.86955027]])
        """

        if by > self.n_sample:
            raise ValueError(f"by argument {by} must be lower or equal to the sample size {self.n_sample}")

        ind_start = 0
        ind_end = by

        while ind_start != ind_end:
            yield self.slice(start=ind_start, end=ind_end)
            ind_start = ind_end
            ind_end = np.minimum(ind_end + by, self.n_sample)

    def to_numpy(self, axis: int = 0) -> np.ndarray:
        """
        Convert the Samples object to a `numpy.ndarray`.

        The attribute arrays are stacked along a user-specified axis of the resulting array.

        Parameters
        ----------
        axis : `int`, default 0
            The axis along which the generated samples of each rainfall-runoff parameter and/or initial state
            will be joined.

        Returns
        -------
        res : `numpy.ndarray`
            It returns the `Samples` object as a `numpy.ndarray`.

        Examples
        --------
        >>> from smash.factory import generate_samples
        >>> problem = {"num_vars": 2, "names": ["cp", "llr"], "bounds": [[1, 200], [1, 500]]}

        Generate samples

        >>> sample = generate_samples(problem, n=5, random_state=1)

        Convert the result to a `numpy.ndarray`

        >>> sample.to_numpy(axis=-1)
        array([[ 83.98737894,  47.07695879],
               [144.34457419,  93.94384548],
               [  1.02276059, 173.43480279],
               [ 61.16418195, 198.98696964],
               [ 30.20442227, 269.86955027]])
        """

        return np.stack([getattr(self, k) for k in self._problem["names"]], axis=axis)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the Samples object to a `pandas.DataFrame`.

        Returns
        -------
        res : `pandas.DataFrame`
            It returns the `Samples` object as a `pandas.DataFrame`.

        Examples
        --------
        >>> from smash.factory import generate_samples
        >>> problem = {"num_vars": 2, "names": ["cp", "llr"], "bounds": [[1, 200], [1, 500]]}

        Generate samples

        >>> sample = generate_samples(problem, n=5, random_state=1)

        Convert the result to a `pandas.DataFrame`

        >>> sample.to_dataframe()
                   cp         llr
        0   83.987379   47.076959
        1  144.344574   93.943845
        2    1.022761  173.434803
        3   61.164182  198.986970
        4   30.204422  269.869550
        """

        return pd.DataFrame({k: getattr(self, k) for k in self._problem["names"]})


def generate_samples(
    problem: dict[str, Any],
    generator: str = "uniform",
    n: Numeric = 1000,
    random_state: Numeric | None = None,
    mean: dict[str, float] | None = None,
    coef_std: Numeric | None = None,
) -> Samples:
    """
    Generate a multiple set of variables.

    Parameters
    ----------
    problem : `dict[str, Any]`
        Problem definition. The keys are

        - ``'num_vars'`` : the number of variables.
        - ``'names'`` : the name of the variables.
        - ``'bounds'`` : the upper and lower bounds of each variable (a sequence of ``(min, max)``).

    generator : `str`, default 'uniform'
        Samples generator. Should be one of

        - ``'uniform'``
        - ``'normal'`` (or ``'gaussian'``)

    n : `int`, default 1000
        Number of generated samples.

    random_state : `int` or None, default None
        Random seed used to generate samples.

        .. note::
            If not given, generates parameters sets with a random seed.

    mean : `dict[str, float]` or None, default None
        If the samples are generated using a Gaussian distribution (i.e. ``'normal'`` or ``'gaussian'`` in
        **generator**), **mean** is used to define the mean of the distribution for each variable.
        It is a dictionary where keys are the name of the rainfall-runoff parameters and/or initial states
        defined in the **problem** argument. In this case, the truncated normal distribution may be used with
        respect to the boundary conditions defined in **problem**. None value inside the dictionary will be
        filled in with the center of the rainfall-runoff parameter and/or initial state bounds.

        .. note::
            If not given and Gaussian distribution is used, the mean of the distribution will be set to the
            center of the variable bounds.

    coef_std : `float` or None, default None
        A coefficient related to the standard deviation in case of Gaussian generator:

        .. math::
                std = \\frac{u - l}{coef\\_std}

        where :math:`u` and :math:`l` are the upper and lower bounds of variables.

        .. note::
            If not given and Gaussian distribution is used, **coef_std** is set to 3 as default:

            .. math::
                std = \\frac{u - l}{3}

    Returns
    -------
    samples : `Samples <smash.Samples>`
        It returns an object containing the generated samples result.

    See Also
    --------
    smash.Samples: Represents the generated samples result.

    Examples
    --------
    >>> from smash.factory import generate_samples

    Define the problem by a dictionary

    >>> problem = {
        'num_vars': 4,
        'names': ['cp', 'ct', 'kexc', 'llr'],
        'bounds': [[1, 2000], [1, 1000], [-20, 5], [1, 1000]]
    }

    Generate samples

    >>> sr = generate_samples(problem, n=3, random_state=99)

    Convert sample to `pandas.DataFrame`

    >>> sr.to_dataframe()
                cp          ct       kexc         llr
    0  1344.884839   32.414941 -12.559438    7.818907
    1   976.668720  808.241913 -18.832607  770.023235
    2  1651.164853  566.051802   4.765685  747.020334
    """

    args = _standardize_generate_samples_args(problem, generator, n, random_state, mean, coef_std)

    return _generate_samples(*args)


def _generate_samples(
    problem: dict,
    generator: str,
    n: int,
    random_state: int | None,
    mean: dict | None,
    coef_std: Numeric | None,
) -> Samples:
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

            ret_dict["_dst_" + p] = np.ones(n) / (upp - low)

        elif generator in ["normal", "gaussian"]:
            sd = (upp - low) / coef_std

            trunc_normal = truncnorm((low - mean[p]) / sd, (upp - mean[p]) / sd, loc=mean[p], scale=sd)

            ret_dict[p] = trunc_normal.rvs(size=n)

            ret_dict["_dst_" + p] = trunc_normal.pdf(ret_dict[p])

    # % Reset random seed if random_state is previously set
    if random_state is not None:
        np.random.seed(None)

    return Samples(ret_dict)
