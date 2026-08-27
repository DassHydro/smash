from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from smash._constant import (
    SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS,
    STRUCTURE_RR_INTERNAL_FLUXES,
)
from smash.core.model._build_model import _map_dict_to_fortran_derived_type
from smash.core.simulation._doc import (
    _backward_run_doc_appender,
    _forward_run_doc_appender,
    _multiple_forward_run_doc_appender,
    _smash_backward_run_doc_substitution,
    _smash_forward_run_doc_substitution,
)
from smash.core.simulation.optimize._tools import _get_parameters_b, _get_parameters_q_b
from smash.core.simulation.run._standardize import (
    _standardize_multiple_forward_run_args,
)
from smash.fcore._mw_forward import (
    forward_run as wrap_forward_run,
)
from smash.fcore._mw_forward import (
    multiple_forward_run as wrap_multiple_forward_run,
)
from smash.fcore._mwd_options import OptionsDT
from smash.fcore._mwd_parameters_manipulation import (
    parameters_to_control as wrap_parameters_to_control,
)
from smash.fcore._mwd_returns import ReturnsDT

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from smash.core.model.model import Model
    from smash.factory.samples.samples import Samples
    from smash.util._typing import Numeric

__all__ = [
    "BackwardRun",
    "ForwardRun",
    "MultipleForwardRun",
    "backward_run",
    "forward_run",
    "multiple_forward_run",
]


class MultipleForwardRun:
    """
    Represents multiple forward run result.

    Attributes
    ----------
    cost : `numpy.ndarray`
        An array of shape *(n,)* representing cost values from *n* simulations.

    q : `numpy.ndarray`
        An array of shape *(ng, ntime_step, n)* representing simulated discharges from *n* simulations.

    See Also
    --------
    multiple_forward_run : Run the forward Model with multiple sets of parameters.
    """

    def __init__(self, data: dict[str, NDArray[np.float32]] | None = None):
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


class ForwardRun:
    """
    Represents forward run optional results.

    Attributes
    ----------
    time_step : `pandas.DatetimeIndex`
        A list of length *n* containing the returned time steps.

    rr_states : `FortranDerivedTypeArray`
        A list of length *n* of `RR_StatesDT <fcore._mwd_rr_states.RR_StatesDT>` for each **time_step**.

    q_domain : `numpy.ndarray`
        An array of shape *(nrow, ncol, n)* representing simulated discharges on the domain for each
        **time_step**.

    internal_fluxes : dict[str, `numpy.ndarray`]
        A dictionary where keys are the names of the internal fluxes and the values are array of
        shape *(nrow, ncol, n)* representing an internal flux on the domain for each **time_step**.

    cost : `float`
        Cost value.

    jobs : `float`
        Cost observation component value.

    Notes
    -----
    The object's available attributes depend on what is requested by the user in **return_options** during a
    call to `forward_run`.

    See Also
    --------
    smash.forward_run : Run the forward Model.
    """

    def __init__(self, data: dict[str, Any] | None = None):
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


class BackwardRun:
    """
    Represents backward run optional results.

    Attributes
    ----------
    time_step : `pandas.DatetimeIndex`
        A list of length *n* containing the returned time steps.

    rr_states : `FortranDerivedTypeArray`
        A list of length *n* of `RR_StatesDT <fcore._mwd_rr_states.RR_StatesDT>` for each **time_step**.

    q_domain : `numpy.ndarray`
        An array of shape *(nrow, ncol, n)* representing simulated discharges on the domain for each
        **time_step**.

    internal_fluxes : dict[str, `numpy.ndarray`]
        A dictionary where keys are the names of the internal fluxes and the values are array of
        shape *(nrow, ncol, n)* representing an internal flux on the domain for each **time_step**.

    cost : `float`
        Cost value.

    jobs : `float`
        Cost observation component value.

    grad : `numpy.ndarray`
        An array of shape *(n,)* representing the gradient value.

    Notes
    -----
    The object's available attributes depend on what is requested by the user in **return_options** during a
    call to `backward_run`.

    See Also
    --------
    smash.backward_run : Run the backward Model.
    """

    def __init__(self, data: dict[str, Any] | None = None):
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


@_smash_forward_run_doc_substitution
@_forward_run_doc_appender
def forward_run(
    model: Model,
    cost_options: dict[str, Any] | None = None,
    common_options: dict[str, Any] | None = None,
    return_options: dict[str, Any] | None = None,
) -> Model | tuple[Model, ForwardRun]:
    wmodel = model.copy()

    ret_forward_run = wmodel.forward_run(cost_options, common_options, return_options)

    if ret_forward_run is None:
        return wmodel
    else:
        return (wmodel, ret_forward_run)


def _forward_run(
    model: Model, cost_options: dict, common_options: dict, return_options: dict
) -> ForwardRun | None:
    if common_options["verbose"]:
        print("</> Forward Run")

    wrap_options = OptionsDT(
        model.setup,
        model.mesh,
        cost_options["njoc"],
        cost_options["njrc"],
    )

    wrap_returns = ReturnsDT(
        model.setup,
        model.mesh,
        return_options["nmts"],
        return_options["fkeys"],
    )

    # % Map cost_options dict to derived type
    _map_dict_to_fortran_derived_type(cost_options, wrap_options.cost)

    # % Map common_options dict to derived type
    _map_dict_to_fortran_derived_type(common_options, wrap_options.comm)

    # % Map return_options dict to derived type
    _map_dict_to_fortran_derived_type(return_options, wrap_returns)

    wrap_forward_run(
        model.setup,
        model.mesh,
        model._input_data,
        model._parameters,
        model._output,
        wrap_options,
        wrap_returns,
    )

    fret = {}
    pyret = {}

    for key in return_options["keys"]:
        try:
            value = getattr(wrap_returns, key)
        except Exception:
            continue
        if hasattr(value, "copy"):
            value = value.copy()
        fret[key] = value

    ret = {**fret, **pyret}

    if ret:
        if "internal_fluxes" in ret:
            ret["internal_fluxes"] = {
                key: ret["internal_fluxes"][..., i]
                for i, key in enumerate(STRUCTURE_RR_INTERNAL_FLUXES[model.setup.structure])
            }

        # % Add time_step to the object
        if any(k in SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS for k in ret.keys()):
            ret["time_step"] = return_options["time_step"].copy()
        return ForwardRun(ret)


@_smash_backward_run_doc_substitution
@_backward_run_doc_appender
def backward_run(
    model: Model,
    diff_target: str = "j",
    cotangent: Numeric | NDArray[np.float32] | None = None,
    mapping: str = "uniform",
    optimizer: str | None = None,
    optimize_options: dict[str, Any] | None = None,
    cost_options: dict[str, Any] | None = None,
    common_options: dict[str, Any] | None = None,
    return_options: dict[str, Any] | None = None,
) -> Model | tuple[Model, BackwardRun]:
    wmodel = model.copy()

    ret_backward_run = wmodel.backward_run(
        diff_target,
        cotangent,
        mapping,
        optimizer,
        optimize_options,
        cost_options,
        common_options,
        return_options,
    )

    if ret_backward_run is None:
        return wmodel
    else:
        return (wmodel, ret_backward_run)


def _backward_run(
    model: Model,
    diff_target: str,
    cotangent: np.ndarray,
    mapping: str,
    optimizer: str,
    optimize_options: dict,
    cost_options: dict,
    common_options: dict,
    return_options: dict,
) -> BackwardRun | None:
    if common_options["verbose"]:
        print("</> Backward Run")

    wrap_options = OptionsDT(
        model.setup,
        model.mesh,
        cost_options["njoc"],
        cost_options["njrc"],
    )

    wrap_returns = ReturnsDT(
        model.setup,
        model.mesh,
        return_options["nmts"],
        return_options["fkeys"],
    )

    # % Map optimize_options dict to derived type
    _map_dict_to_fortran_derived_type(optimize_options, wrap_options.optimize)

    # % Map cost_options dict to derived type
    _map_dict_to_fortran_derived_type(cost_options, wrap_options.cost)

    # % Map common_options dict to derived type
    _map_dict_to_fortran_derived_type(common_options, wrap_options.comm)

    # % Map return_options dict to derived type
    _map_dict_to_fortran_derived_type(return_options, wrap_returns)

    wparameters = model._parameters.copy()

    wrap_parameters_to_control(
        model.setup,
        model.mesh,
        model._input_data,
        wparameters,
        wrap_options,
    )

    if diff_target == "j":
        parameters_b = _get_parameters_b(model, wparameters, wrap_options, wrap_returns, cotangent)
    elif diff_target == "q":
        parameters_b = _get_parameters_q_b(model, wparameters, wrap_options, wrap_returns, cotangent)
    else:  # Should be unreachable
        pass

    grad = parameters_b.control.x.copy()

    wrap_forward_run(
        model.setup,
        model.mesh,
        model._input_data,
        model._parameters,
        model._output,
        wrap_options,
        wrap_returns,
    )

    fret = {}
    pyret = {}

    for key in return_options["keys"]:
        try:
            value = getattr(wrap_returns, key)
        except Exception:
            continue
        if hasattr(value, "copy"):
            value = value.copy()
        fret[key] = value

    if "grad" in return_options["keys"]:
        pyret["grad"] = grad

    ret = {**fret, **pyret}

    if ret:
        if "internal_fluxes" in ret:
            ret["internal_fluxes"] = {
                key: ret["internal_fluxes"][..., i]
                for i, key in enumerate(STRUCTURE_RR_INTERNAL_FLUXES[model.setup.structure])
            }

        # % Add time_step to the object
        if any(k in SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS for k in ret.keys()):
            ret["time_step"] = return_options["time_step"].copy()
        return BackwardRun(ret)


@_multiple_forward_run_doc_appender
def multiple_forward_run(
    model: Model,
    samples: Samples | dict[str, Any],
    cost_options: dict[str, Any] | None = None,
    common_options: dict[str, Any] | None = None,
) -> MultipleForwardRun:
    args_options = [deepcopy(arg) for arg in [cost_options, common_options]]

    args = _standardize_multiple_forward_run_args(model, samples, *args_options)

    res = _multiple_forward_run(model, *args)

    return MultipleForwardRun(res)


def _multiple_forward_run(
    model: Model,
    samples: Samples | None,
    spatialized_samples: dict[str, np.ndarray],
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
    _map_dict_to_fortran_derived_type(cost_options, wrap_options.cost)

    # % Map common_options dict to derived type
    _map_dict_to_fortran_derived_type(common_options, wrap_options.comm)

    # % Generate samples info
    nv = len(spatialized_samples)
    samples_kind = np.zeros(shape=nv, dtype=np.int32, order="F")
    samples_ind = np.zeros(shape=nv, dtype=np.int32, order="F")

    for i, name in enumerate(spatialized_samples.keys()):
        if name in model._parameters.rr_parameters.keys:
            samples_kind[i] = 0
            # % Adding 1 because Fortran uses one based indexing
            samples_ind[i] = np.argwhere(model._parameters.rr_parameters.keys == name).item() + 1
        elif name in model._parameters.rr_initial_states.keys:
            samples_kind[i] = 1
            # % Adding 1 because Fortran uses one based indexing
            samples_ind[i] = np.argwhere(model._parameters.rr_initial_states.keys == name).item() + 1
        # % Should be unreachable
        else:
            pass

    # % Initialise results
    n_sample = next(iter(spatialized_samples.values())).shape[-1]
    cost = np.zeros(shape=n_sample, dtype=np.float32, order="F")
    q = np.zeros(
        shape=(*model.response_data.q.shape, n_sample),
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
        np.transpose(list(spatialized_samples.values()), (1, 2, 0, 3)),
        samples_kind,
        samples_ind,
        cost,
        q,
    )

    return {
        "cost": cost,
        "q": q,
        "_samples": samples,
        "_spatialized_samples": spatialized_samples,
        "_cost_options": cost_options,
    }
