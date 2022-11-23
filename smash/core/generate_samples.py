from __future__ import annotations

import numpy as np
import pandas as pd
import sobol as sb
from SALib.sample import saltelli
from scipy.stats import truncnorm
from smash.core._optimize import STRUCTURE_PARAMETERS

__all__ = ["generate_samples"]


def _get_truncated_normal(mean, sd, low, upp):
        
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def generate_samples(problem, 
                     generator="uniform", 
                     n=1000, 
                     random_state=None,
                     backg_sol=None,
                     coef_std=None,
                    ):
    """
    Generate a multiple set of spatially uniform distributed model parameters.

    Parameters
    ----------
    problem : dict or Model
            Problem definition.

    generator : str, default uniform
            Samples generator.
            Should be one of

            - 'uniform'
            - 'normal' or 'gaussian'
            - 'sobol'
            - 'saltelli'

    n : int, default 1000
            Number of generated samples.
            In case of Saltelli generator, this is the number of trajectories to generate for each model parameter (ideally a power of 2).
            Then the number of sample to generate for all model parameters is equal to :math:`N(2D+2)` 
            where :math:`D` is the number of model parameters.
            
            See `here <https://salib.readthedocs.io/en/latest/api.html>`__ for more details.

    random_state : int or None, default None
            Random seed used to generate sample, except Sobol and Saltelli, which are determinist generators 
            and do not require a random seed.

            If None, generate parameters sets with a random seed.

    backg_sol : ndarray or None, default None
            Prior solutions could be included in parameters sets, except Sobol and Saltelli generator, and are 
            used as the mean when generating with Gaussian distribution.

            If None, the mean is the center of the parameter bound if in case of Gaussian generator, otherwise, 
            there is no background solution included in the generated parameter sets.

    coef_std : float or None
            A coefficient related to the standard deviation in case of Gaussian generator:

            .. math::
                    std = \\frac{upper - lower}{coef\_std}
            
            If None, a default value for this coefficient will be assigned to define the standard deviation:

            .. math::
                    std = \\frac{upper - lower}{3}

    Returns
    -------
    df : DataFrame
            `df` with all generated samples

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

    if generator == "sobol":  # determinist generator
        sample = sb.sample(dimension=problem["num_vars"], n_points=n)

        for flag, parameter in enumerate(problem["names"]):
            df[parameter] = (
                sample[:, flag]
                * (problem["bounds"][flag][1] - problem["bounds"][flag][0])
                + problem["bounds"][flag][0]
            )

    elif generator == "saltelli": # determinist generator
  
        sample = saltelli.sample(problem, n)

        df[df.keys()] = sample

    else: # non-determinist generator
        if random_state is not None:
            np.random.seed(random_state)

        for i,p in enumerate(problem['names']):

            low = problem['bounds'][i][0]
            upp = problem['bounds'][i][1]

            if backg_sol == None:
                    ubi = []

            else:
                    ubi = [backg_sol[i]]

            if generator == 'uniform':

                    df[p] = np.append(ubi, np.random.uniform(low, upp, n-len(ubi)))

            elif generator == 'normal' or generator == 'gaussian':

                    if coef_std == None:
                            sd = (upp-low)/3

                    else:
                            sd = (upp-low)/coef_std

                    if backg_sol == None:
                            trunc_normal = _get_truncated_normal((low+upp)/2, sd, low, upp)
                    
                    else:
                            trunc_normal = _get_truncated_normal(ubi[0], sd, low, upp)

                    df[p] = np.append(ubi, trunc_normal.rvs(size=n-len(ubi)))

            else:
                raise ValueError(f"Unknown generator {generator}!")

    return df

def _model2problem(modelsetup):

        control_vector = STRUCTURE_PARAMETERS[modelsetup.structure]

        bounds = []

        for name in control_vector:

                if name in modelsetup._parameters_name:

                        ind = np.argwhere(modelsetup._parameters_name == name)

                        l = modelsetup._optimize.lb_parameters[ind].item()
                        u = modelsetup._optimize.ub_parameters[ind].item()

                        bounds += [[l,u]]

        problem = {
                'num_vars': len(control_vector),
                'names': control_vector,
                'bounds': bounds
                }

        return problem
