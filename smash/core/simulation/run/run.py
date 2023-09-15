from __future__ import annotations

from smash._constant import (
    SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS,
)

from smash.core.model._build_model import _map_dict_to_object

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

__all__ = ["MultipleForwardRun", "ForwardRun", "forward_run", "multiple_forward_run"]


class MultipleForwardRun:
    """
    Represents multiple forward run computation result.

    Attributes
    ----------
    cost : numpy.ndarray
        An array of shape *(n,)* representing cost values from *n* simulations.

    q : numpy.ndarray
        An array of shape *(..., n)* representing simulated discharges from *n* simulations.

    See Also
    --------
    multiple_forward_run : Run the forward Model with multiple sets of parameters.
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
                [
                    k.rjust(m) + ": " + repr(type(v))
                    for k, v in sorted(dct.items())
                    if not k.startswith("_")
                ]
            )
        else:
            return self.__class__.__name__ + "()"


class ForwardRun:
    """
    Represents forward run optional results.

    Attributes
    ----------
    time_step : pandas.DatetimeIndex
        A pandas.DatetimeIndex containing *n* returned time steps.

    opr_states : list
        A list of length *n* of Opr_StatesDT for each **time_step**.

    q_domain : numpy.ndarray
        An array of shape *(nrow, ncol, n)* representing simulated discharges on the domain for each **time_step**.

    cost : float
        Cost value.

    jobs : float
        Cost observation component value.

    Notes
    -----
    The object's available attributes depend on what is requested by the user in **return_options** during a call to `smash.forward_run`.

    See Also
    --------
    smash.forward_run : Run the forward Model.
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
                [
                    k.rjust(m) + ": " + repr(type(v))
                    for k, v in sorted(dct.items())
                    if not k.startswith("_")
                ]
            )
        else:
            return self.__class__.__name__ + "()"


def forward_run(
    model: Model,
    cost_options: dict | None = None,
    common_options: dict | None = None,
    return_options: dict | None = None,
) -> Model | (Model, ForwardRun):
    """
    Run the forward Model.

    Parameters
    ----------
    model : Model
        Model object.

    cost_options : dict or None, default None
        Dictionary containing computation cost options for simulated and observed responses. The elements are:

        jobs_cmpt : str or ListLike, default 'nse'
            Type of observation objective function(s) to be computed. Should be one or a sequence of any of

            - 'nse', 'nnse', 'kge', 'mae', 'mape', 'mse', 'rmse', 'lgrm' (classical evaluation metrics)
            - 'Crc', 'Crchf', 'Crclf', 'Crch2r', 'Cfp2', 'Cfp10', 'Cfp50', 'Cfp90' (continuous signatures-based error metrics)
            - 'Eff', 'Ebf', 'Erc', 'Erchf', 'Erclf', 'Erch2r', 'Elt', 'Epf' (flood event signatures-based error metrics)

            .. hint::
                See a detailed explanation on the objective function in :ref:`Math / Num Documentation <math_num_documentation.signal_analysis.cost_functions>` section.

        wjobs_cmpt : str, Numeric, or ListLike, default 'mean'
            The corresponding weighting of observation objective functions in case of multi-criteria (i.e., a sequence of objective functions to compute). The default is set to the average weighting.

        gauge : str or ListLike, default 'dws'
            Type of gauge to be computed. There are two ways to specify it:

            - A gauge code or any sequence of gauge codes. The gauge code(s) given must belong to the gauge codes defined in the Model mesh.
            - An alias among 'all' (all gauge codes) and 'dws' (most downstream gauge code(s)).

        wgauge : str or ListLike, default 'mean'
            Type of gauge weights. There are two ways to specify it:

            - A sequence of value whose size must be equal to the number of gauges optimized.
            - An alias among 'mean', 'lquartile' (1st quantile or lower quantile), 'median', or 'uquartile' (3rd quantile or upper quantile).

        event_seg : dict, default {'peak_quant': 0.995, 'max_duration': 240}
            A dictionary of event segmentation options when calculating flood event signatures for cost computation (i.e., **jobs_cmpt** includes flood events signatures).
            See `smash.hydrograph_segmentation` for more.

        end_warmup : str or pandas.Timestamp, default model.setup.start_time
            The end of the warm-up period, which must be between the start time and the end time defined in the Model setup. By default, it is set to be equal to the start time.

        .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

    common_options : dict or None, default None
        Dictionary containing common options with two elements:

        verbose : bool, default False
            Whether to display information about the running method.

        ncpu : bool, default 1
            Whether to perform a parallel computation.

        .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

    return_options : dict or None, default None
        Dictionary containing return options to save intermediate variables. The elements are:

        time_step : str, pandas.Timestamp, pandas.DatetimeIndex or ListLike, default 'all'
            Returned time steps. There are five ways to specify it:

            - A date as a character string which respect pandas.Timestamp format (i.e. '1997-12-21', '19971221', ...).
            - An alias among 'all' (return all time steps).
            - A pandas.Timestamp object.
            - A pandas.DatetimeIndex object.
            - A sequence of dates as character string or pandas.Timestamp (i.e. ['1998-23-05', '1998-23-06'])

            .. note::
                It only applies to the following variables: 'opr_states' and 'q_domain'

        opr_states : bool, default False
            Whether to return operator states for specific time steps.

        q_domain : bool, defaul False
            Whether to return simulated discharge on the whole domain for specific time steps.

        cost : bool, default False
            Whether to return cost value.

        jobs : bool, default False
            Whether to return jobs (observation component of cost) value.

        .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

    Returns
    -------
    ret_model : Model
        The Model with forward run outputs.

    ret_forward_run : ForwardRun or None, default None
        It returns a `smash.ForwardRun` object containing the intermediate variables defined in **return_options**. If no intermediate variables are defined, it returns None.

    Examples
    --------
    TODO: Fill

    See Also
    --------
    Model.forward_run : Run the forward Model.
    smash.ForwardRun : Represents forward run optional results.
    """

    wmodel = model.copy()

    ret_forward_run = wmodel.forward_run(cost_options, common_options, return_options)

    if ret_forward_run is None:
        return wmodel
    else:
        return wmodel, ret_forward_run


def _forward_run(
    model: Model, cost_options: dict, common_options: dict, return_options: dict
) -> ForwardRunReturns | None:
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
    _map_dict_to_object(cost_options, wrap_options.cost)

    # % Map common_options dict to derived type
    _map_dict_to_object(common_options, wrap_options.comm)

    # % Map return_options dict to derived type
    _map_dict_to_object(return_options, wrap_returns)

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
        except:
            continue
        if hasattr(value, "copy"):
            value = value.copy()
        fret[key] = value

    ret = {**fret, **pyret}
    if ret:
        # % Add time_step to the object
        if any([k in SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS for k in ret.keys()]):
            ret["time_step"] = return_options["time_step"].copy()
        return ForwardRun(ret)


def multiple_forward_run(
    model: Model,
    samples: Samples,
    cost_options: dict | None = None,
    common_options: dict | None = None,
) -> MultipleForwardRun:
    """
    Run the forward Model with multiple sets of parameters.

    Parameters
    ----------
    model : Model
        Model object.

    samples : Samples
        The samples created by the `smash.factory.generate_samples` method.

    cost_options, common_options : multiple types
        Forward run settings. Refer to `smash.forward_run` or `Model.forward_run` for details on these arguments.

    Returns
    -------
    mfr : MultipleForwardRun
        The multiple forward run results represented as a `MultipleForwardRun` object.

    Examples
    --------
    TODO: Fill

    See Also
    --------
    Samples : Represents the generated samples result.
    """

    args_options = [deepcopy(arg) for arg in [cost_options, common_options]]

    args = _standardize_multiple_forward_run_args(model, samples, *args_options)

    res = _multiple_forward_run(model, *args)

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
    _map_dict_to_object(cost_options, wrap_options.cost)

    # % Map common_options dict to derived type
    _map_dict_to_object(common_options, wrap_options.comm)

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

    return {"cost": cost, "q": q, "_samples": samples, "_cost_options": cost_options}
