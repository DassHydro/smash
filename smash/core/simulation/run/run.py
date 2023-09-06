from __future__ import annotations

from smash.core.simulation.run._standardize import (
    _standardize_multiple_forward_run_args,
)

from smash.fcore._mw_forward import (
    forward_run as wrap_forward_run,
    multiple_forward_run as wrap_multiple_forward_run,
)
from smash.fcore._mwd_options import OptionsDT
from smash.fcore._mwd_returns import ReturnsDT

import numpy as np
from copy import deepcopy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.factory.samples.samples import Samples

__all__ = ["MultipleForwardRun", "forward_run", "multiple_forward_run"]


class MultipleForwardRun(dict):
    """
    Represents multiple forward run computation result.

    Notes
    -----
    This class is essentially a subclass of dict with attribute accessors.

    TODO FC: Fill

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
                [
                    k.rjust(m) + ": " + repr(v)
                    for k, v in sorted(self.items())
                    if not k.startswith("_")
                ]
            )
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def forward_run(
    model: Model,
    cost_options: dict | None = None,
    common_options: dict | None = None,
) -> Model:
    
    """
    Run the forward Model.

    TODO: Fill

    Returns
    -------
    ret_model : Model
        The Model with forward run outputs.
    """

    wmodel = model.copy()

    wmodel.forward_run(cost_options, common_options)

    return wmodel


def _forward_run(model: Model, cost_options: dict, common_options: dict):
    if common_options["verbose"]:
        print("</> Forward Run")

    wrap_options = OptionsDT(
        model.setup,
        model.mesh,
        cost_options["njoc"],
        cost_options["njrc"],
    )

    # % Map cost_options dict to derived type
    for key, value in cost_options.items():
        if hasattr(wrap_options.cost, key):
            setattr(wrap_options.cost, key, value)

    # % Map common_options dict to derived type
    for key, value in common_options.items():
        if hasattr(wrap_options.comm, key):
            setattr(wrap_options.comm, key, value)

    # % TODO: Implement return options
    wrap_returns = ReturnsDT()

    wrap_forward_run(
        model.setup,
        model.mesh,
        model._input_data,
        model._parameters,
        model._output,
        wrap_options,
        wrap_returns,
    )


def multiple_forward_run(
    model: Model,
    samples: Samples,
    cost_options: dict | None = None,
    common_options: dict | None = None,
) -> MultipleForwardRun:
    
    """
    Run the forward Model on multiple sets of operator parameters or/and initial states.

    TODO: Fill

    Returns
    -------
    mfr : MultipleForwardRun
        The multiple forward run results represented as a `MultipleForwardRun` object.
    """
    
    args_options = [deepcopy(arg) for arg in [cost_options, common_options]]

    args = _standardize_multiple_forward_run_args(model, samples, *args_options)

    res = _multiple_forward_run(model, *args)

    # % Backup cost options info
    res["_cost_options"] = cost_options

    return MultipleForwardRun(res)


def _multiple_forward_run(
    model: Model,
    samples: Samples,
    cost_options: dict,
    common_options: dict,
) -> dict:
    if common_options["verbose"]:
        print("</> Multiple Forward Run")

    wrap_options = OptionsDT(
        model.setup,
        model.mesh,
        cost_options["njoc"],
        cost_options["njrc"],
    )

    # % Map cost_options dict to derived type
    for key, value in cost_options.items():
        if hasattr(wrap_options.cost, key):
            setattr(wrap_options.cost, key, value)

    # % Map common_options dict to derived type
    for key, value in common_options.items():
        if hasattr(wrap_options.comm, key):
            setattr(wrap_options.comm, key, value)

    # % Generate samples info
    nv = samples._problem["num_vars"]
    samples_kind = np.zeros(shape=nv, dtype=np.int32, order="F")
    samples_ind = np.zeros(shape=nv, dtype=np.int32, order="F")

    for i, name in enumerate(samples._problem["names"]):
        if name in model._parameters.opr_parameters.keys:
            samples_kind[i] = 0
            # % Adding 1 because Fortran uses one based indexing
            samples_ind[i] = (
                np.argwhere(model._parameters.opr_parameters.keys == name).item() + 1
            )
        elif name in model._parameters.opr_initial_states.keys:
            samples_kind[i] = 1
            # % Adding 1 because Fortran uses one based indexing
            samples_ind[i] = (
                np.argwhere(model._parameters.opr_initial_states.keys == name).item()
                + 1
            )
        # % Should be unreachable
        else:
            pass

    # % Initialise results
    cost = np.zeros(shape=samples.n_sample, dtype=np.float32, order="F")
    q = np.zeros(
        shape=(*model.obs_response.q.shape, samples.n_sample),
        dtype=np.float32,
        order="F",
    )

    wrap_multiple_forward_run(
        model.setup,
        model.mesh,
        model._input_data,
        model._parameters,
        model._output,
        wrap_options,
        samples.to_numpy(),
        samples_kind,
        samples_ind,
        cost,
        q,
    )

    return {"cost": cost, "q": q, "_samples": samples}
