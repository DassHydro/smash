from __future__ import annotations

from smash._constant import SAMPLE_GENERATORS, PROBLEM_KEYS

from smash.tools._common_function import _default_bound_constraints

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver._mwd_setup import SetupDT

import warnings

import numpy as np


def _standardize_problem(problem: dict | None, setup: SetupDT, states: bool):
    if problem is None:
        problem = _default_bound_constraints(setup, states)

    elif isinstance(problem, dict):
        prl_keys = problem.keys()

        if not all(k in prl_keys for k in PROBLEM_KEYS):
            raise KeyError(
                f"Problem dictionary should be defined with required keys {PROBLEM_KEYS}"
            )

        unk_keys = [k for k in prl_keys if k not in PROBLEM_KEYS]

        if unk_keys:
            warnings.warn(
                f"Unknown key(s) found in the problem definition {unk_keys}. Choices: {PROBLEM_KEYS}"
            )

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
                            f"Key '{name}' does not match any existing names in the problem definition {problem['names']}"
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
