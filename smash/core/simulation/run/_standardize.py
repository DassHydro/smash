from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from smash.core.model._standardize import _standardize_rr_parameters_value, _standardize_rr_states_value
from smash.core.simulation._standardize import (
    _standardize_simulation_common_options,
    _standardize_simulation_cost_options,
    _standardize_simulation_cost_options_finalize,
    _standardize_simulation_parameters_feasibility,
    _standardize_simulation_return_options,
    _standardize_simulation_return_options_finalize,
    _standardize_simulation_samples,
)
from smash.factory.samples.samples import Samples

if TYPE_CHECKING:
    from typing import Any

    from smash.core.model.model import Model
    from smash.util._typing import AnyTuple


def _standardize_forward_run_args(
    model: Model,
    cost_options: dict | None,
    common_options: dict | None,
    return_options: dict | None,
) -> AnyTuple:
    func_name = "forward_run"
    # % In case model.set_rr_parameters or model.set_rr_initial_states were not used
    _standardize_simulation_parameters_feasibility(model)

    cost_options = _standardize_simulation_cost_options(model, func_name, cost_options)

    # % Finalize cost_options
    _standardize_simulation_cost_options_finalize(model, func_name, cost_options)

    common_options = _standardize_simulation_common_options(common_options)

    return_options = _standardize_simulation_return_options(model, func_name, return_options)

    # % Finalize return_options
    _standardize_simulation_return_options_finalize(model, return_options)

    return (cost_options, common_options, return_options)


def _standardize_multiple_forward_run_args(
    model: Model,
    samples: Samples | dict,
    cost_options: dict | None,
    common_options: dict | None,
) -> AnyTuple:
    samples, spatialized_samples = _standardize_multiple_forward_run_samples(model, samples)

    # % In case model.set_rr_parameters or model.set_rr_initial_states were not used
    _standardize_simulation_parameters_feasibility(model)

    cost_options = _standardize_simulation_cost_options(model, "forward_run", cost_options)

    # % Finalize cost_options
    _standardize_simulation_cost_options_finalize(model, "forward_run", cost_options)

    common_options = _standardize_simulation_common_options(common_options)

    return (samples, spatialized_samples, cost_options, common_options)


def _standardize_multiple_forward_run_samples(
    model: Model, samples: Samples | dict[str, Any]
) -> tuple[Samples | None, dict[str, np.ndarray]]:
    if isinstance(samples, Samples):
        uniform_samples = _standardize_simulation_samples(model, samples)
        spatialized_samples = {
            k: np.stack(
                [v * np.ones(model.mesh.flwdir.shape, dtype=np.float32) for v in getattr(samples, k)], axis=-1
            )
            for k in samples._problem["names"]
        }

    elif isinstance(samples, dict):
        uniform_samples = None

        try:
            lengths = [len(v) for v in samples.values()]
            if not lengths:
                raise ValueError("samples argument cannot be empty") from None

        except Exception:
            raise ValueError("samples argument cannot contain non-iterable elements") from None

        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All elements in the samples dictionary must have the same length")

        spatialized_samples = {}

        for key, value in samples.items():
            spatialized_samples[key] = np.ones((*model.mesh.flwdir.shape, len(value)), dtype=np.float32)

            if key in model.rr_parameters.keys:
                for i, v in enumerate(value):
                    v = _standardize_rr_parameters_value(model, key, v)
                    spatialized_samples[key][..., i] = v

            elif key in model.rr_initial_states.keys:
                for i, v in enumerate(value):
                    v = _standardize_rr_states_value(model, "rr_initial_state", key, v)
                    spatialized_samples[key][..., i] = v

            else:
                raise ValueError(
                    f"Unknown model rr_parameter or rr_initial_states '{key}'. "
                    f"Choices: {list(model.rr_parameters.keys) + list(model.rr_initial_states.keys)}"
                )

    else:
        raise TypeError("samples argument must a Samples object or dictionary")

    return uniform_samples, spatialized_samples
