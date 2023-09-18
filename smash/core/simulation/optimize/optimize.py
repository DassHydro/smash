from __future__ import annotations

from smash._constant import (
    SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS,
)

from smash.core.model._build_model import _map_dict_to_object

from smash.core.simulation.optimize._standardize import (
    _standardize_multiple_optimize_args,
)

from smash.fcore._mw_forward import forward_run as wrap_forward_run
from smash.fcore._mw_optimize import (
    optimize as wrap_optimize,
    multiple_optimize as wrap_multiple_optimize,
)
from smash.fcore._mwd_options import OptionsDT
from smash.fcore._mwd_returns import ReturnsDT

import numpy as np
from copy import deepcopy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.factory.net.net import Net
    from smash.factory.samples.samples import Samples


__all__ = ["MultipleOptimize", "Optimize", "optimize", "multiple_optimize"]


class MultipleOptimize:
    """
    Represents multiple optimize computation result.

    Attributes
    ----------
    cost : numpy.ndarray
        An array of shape *(n,)* representing cost values from *n* simulations.

    q : numpy.ndarray
        An array of shape *(..., n)* representing simulated discharges from *n* simulations.

    parameters : dict
        A dictionary containing optimized parameters and/or initial states. Each key represents an array of shape *(..., n)* corresponding to a specific parameter or state.

    See Also
    --------
    multiple_optimize : Run multiple optimization processes with different starting points, yielding multiple solutions.
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


class Optimize:
    """
    Represents optimize optional results.

    Attributes
    ----------
    time_step : pandas.DatetimeIndex
        A pandas.DatetimeIndex containing *n* returned time steps.

    opr_states : list
        A list of length *n* of Opr_StatesDT for each **time_step**.

    q_domain : numpy.ndarray
        An array of shape *(nrow, ncol, n)* representing simulated discharges on the domain for each **time_step**.

    iter_cost : numpy.ndarray
        An array of shape *(m,)* representing cost iteration values from *m* iterations.

    iter_projg : numpy.ndarray
        An array of shape *(m,)* representing infinity norm of the projected gardient iteration values from *m* iterations.

    control_vector : numpy.ndarray
        An array of size *(k,)* representing the control vector at end of optimization.

    net : Net
        The trained neural network `smash.factory.Net`.

    cost : float
        Cost value.

    jobs : float
        Cost observation component value.

    jreg : float
        Cost regularization component value.

    Notes
    -----
    The object's available attributes depend on what is requested by the user during a call to `smash.optimize` in **return_options**.

    See Also
    --------
    smash.optimize : Model assimilation using numerical optimization algorithms.
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


def optimize(
    model: Model,
    mapping: str = "uniform",
    optimizer: str | None = None,
    optimize_options: dict | None = None,
    cost_options: dict | None = None,
    common_options: dict | None = None,
    return_options: dict | None = None,
) -> Model | (Model, Optimize):
    """
    Model assimilation using numerical optimization algorithms.

    Parameters
    ----------
    model : Model
        Model object.

    mapping : str, default 'uniform'
        Type of mapping. Should be one of 'uniform', 'distributed', 'multi-linear', 'multi-polynomial', 'ann'.

    optimizer : str or None, default None
        Name of optimizer. Should be one of 'sbs', 'lbfgsb', 'sgd', 'adam', 'adagrad', 'rmsprop'.

        .. note::
            If not given, a default optimizer will be set depending on the optimization mapping:

            - **mapping** = 'uniform'; **optimizer** = 'sbs'
            - **mapping** = 'distributed', 'multi-linear', or 'multi-polynomial'; **optimizer** = 'lbfgsb'
            - **mapping** = 'ann'; **optimizer** = 'adam'

    optimize_options : dict or None, default None
        Dictionary containing optimization options for fine-tuning the optimization process.

        .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element. See the returned parameters in `smash.default_optimize_options` for more.

    cost_options : dict or None, default None
        Dictionary containing computation cost options for simulated and observed responses. The elements are:

        jobs_cmpt : str or ListLike, default 'nse'
            Type of observation objective function(s) to be minimized. Should be one or a sequence of any of

            - 'nse', 'nnse', 'kge', 'mae', 'mape', 'mse', 'rmse', 'lgrm' (classical evaluation metrics)
            - 'Crc', 'Crchf', 'Crclf', 'Crch2r', 'Cfp2', 'Cfp10', 'Cfp50', 'Cfp90' (continuous signatures-based error metrics)
            - 'Eff', 'Ebf', 'Erc', 'Erchf', 'Erclf', 'Erch2r', 'Elt', 'Epf' (flood event signatures-based error metrics)

            .. hint::
                See a detailed explanation on the objective function in :ref:`Math / Num Documentation <math_num_documentation.signal_analysis.cost_functions>` section.

        wjobs_cmpt : str, Numeric, or ListLike, default 'mean'
            The corresponding weighting of observation objective functions in case of multi-criteria (i.e., a sequence of objective functions to compute). The default is set to the average weighting.

        wjreg : Numeric, default 0
            The weighting of regularization term. Only used with distributed mapping.

        jreg_cmpt : str or ListLike, default 'prior'
            Type(s) of regularization function(s) to be minimized when regularization term is set (i.e., **wjreg** > 0). Should be one or a sequence of any of 'prior' and 'smoothing'.

        wjreg_cmpt : str, Numeric, or ListLike, default 'mean'
            The corresponding weighting of regularization functions in case of multi-regularization (i.e., a sequence of regularization functions to compute). The default is set to the average weighting.

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

            - A date as a character string which respect pandas.Timestamp format (i.e., '1997-12-21', '19971221', etc.).
            - An alias among 'all' (return all time steps).
            - A pandas.Timestamp object.
            - A pandas.DatetimeIndex object.
            - A sequence of dates as character string or pandas.Timestamp (i.e., ['1998-05-23', '1998-05-24'])

            .. note::
                It only applies to the following variables: 'opr_states' and 'q_domain'

        opr_states : bool, default False
            Whether to return operator states for specific time steps.

        q_domain : bool, defaul False
            Whether to return simulated discharge on the whole domain for specific time steps.

        iter_cost : bool, default False
            Whether to return cost iteration values.

        iter_projg : bool, default False
            Whether to return infinity norm of the projected gardient iteration values.

        control_vector : bool, default False
            Whether to return control vector at end of optimization. In case of optimization with ANN-based mapping, the control vector is represented in `smash.factory.Net.layers` instead.

        net : Net, default False
            Whether to return the trained neural network `smash.factory.Net`. Only used with ANN-based mapping.

        cost : bool, default False
            Whether to return cost value.

        jobs : bool, default False
            Whether to return jobs (observation component of cost) value.

        jreg : bool, default False
            Whether to return jreg (regularization component of cost) value.

        .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

    Returns
    -------
    ret_model : Model
        The Model with optimized outputs.

    ret_optimize : Optimize or None, default None
        It returns a `smash.Optimize` object containing the intermediate variables defined in **return_options**. If no intermediate variables are defined, it returns None.

    Examples
    --------
    TODO: Fill

    See Also
    --------
    Model.optimize : Model assimilation using numerical optimization algorithms.
    Optimize : Represents optimize optional results.
    """

    wmodel = model.copy()

    ret_optimize = wmodel.optimize(
        mapping,
        optimizer,
        optimize_options,
        cost_options,
        common_options,
        return_options,
    )

    if ret_optimize is None:
        return wmodel
    else:
        return wmodel, ret_optimize


def _optimize(
    model: Model,
    mapping: str,
    optimizer: str,
    optimize_options: dict,
    cost_options: dict,
    common_options: dict,
    return_options: dict,
) -> Optimize | None:
    if common_options["verbose"]:
        print("</> Optimize")

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
    _map_dict_to_object(optimize_options, wrap_options.optimize)

    # % Map cost_options dict to derived type
    _map_dict_to_object(cost_options, wrap_options.cost)

    # % Map common_options dict to derived type
    _map_dict_to_object(common_options, wrap_options.comm)

    # % Map return_options dict to derived type
    _map_dict_to_object(return_options, wrap_returns)

    if mapping == "ann":
        net = _ann_optimize(
            model,
            optimizer,
            optimize_options,
            common_options,
            wrap_options,
            wrap_returns,
        )

    else:
        wrap_optimize(
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

    # % Fortran returns
    for key in return_options["keys"]:
        try:
            value = getattr(wrap_returns, key)
        except:
            continue
        if hasattr(value, "copy"):
            value = value.copy()
        fret[key] = value

    # % Python returns
    if mapping == "ann":
        if "net" in return_options["keys"]:
            pyret["net"] = net
        if "iter_cost" in return_options["keys"]:
            pyret["iter_cost"] = net.history["loss_train"]
        if "iter_projg" in return_options["keys"]:
            pyret["iter_projg"] = net.history["proj_grad"]

    ret = {**fret, **pyret}
    if ret:
        # % Add time_step to the object
        if any([k in SIMULATION_RETURN_OPTIONS_TIME_STEP_KEYS for k in ret.keys()]):
            ret["time_step"] = return_options["time_step"].copy()
        return Optimize(ret)


def _ann_optimize(
    model: Model,
    optimizer: str,
    optimize_options: dict,
    common_options: dict,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
) -> Net:
    # % Preprocessing input descriptors and normalization
    active_mask = np.where(model.mesh.active_cell == 1)

    l_desc = model._input_data.physio_data.l_descriptor
    u_desc = model._input_data.physio_data.u_descriptor

    desc = model._input_data.physio_data.descriptor.copy()
    desc = (desc - l_desc) / (u_desc - l_desc)  # normalize input descriptors

    # % Training the network
    x_train = desc[active_mask]

    net = optimize_options["net"]

    # % Change the mapping to trigger distributed control to get distributed gradients
    wrap_options.optimize.mapping = "distributed"

    net._fit_d2p(
        x_train,
        active_mask,
        model,
        wrap_options,
        wrap_returns,
        optimizer,
        optimize_options["parameters"],
        optimize_options["learning_rate"],
        optimize_options["random_state"],
        optimize_options["termination_crit"]["epochs"],
        optimize_options["termination_crit"]["early_stopping"],
        common_options["verbose"],
    )

    # % Manually deallocate control once fit_d2p done
    model._parameters.control.dealloc()

    # % Reset mapping to ann once fit_d2p done
    wrap_options.optimize.mapping = "ann"

    # % Predicting at active and inactive cells
    x = desc.reshape(-1, desc.shape[-1])
    y = net._predict(x)

    for i, name in enumerate(optimize_options["parameters"]):
        y_reshape = y[:, i].reshape(desc.shape[:2])

        if name in model.opr_parameters.keys:
            ind = np.argwhere(model.opr_parameters.keys == name).item()

            model.opr_parameters.values[..., ind] = y_reshape

        else:
            ind = np.argwhere(model.opr_initial_states.keys == name).item()

            model.opr_inital_states.values[..., ind] = y_reshape

    # % Forward run for updating final states
    wrap_forward_run(
        model.setup,
        model.mesh,
        model._input_data,
        model._parameters,
        model._output,
        wrap_options,
        wrap_returns,
    )

    return net


def multiple_optimize(
    model: Model,
    samples: Samples,
    mapping: str = "uniform",
    optimizer: str | None = None,
    optimize_options: dict | None = None,
    cost_options: dict | None = None,
    common_options: dict | None = None,
) -> MultipleOptimize:
    """
    Run multiple optimization processes with different starting points, yielding multiple solutions.

    Parameters
    ----------
    model : Model
        Model object.

    samples : Samples
        The samples created by the `smash.factory.generate_samples` method.

    mapping, optimizer, optimize_options, cost_options, common_options : multiple types
        Optimization settings. Refer to `smash.optimize` or `Model.optimize` for details on these arguments.

    Returns
    -------
    mopt : MultipleOptimize
        The multiple optimize results represented as a `MultipleOptimize` object.

    Examples
    --------
    TODO: Fill

    See Also
    --------
    Samples : Represents the generated samples result.
    """

    args_options = [
        deepcopy(arg) for arg in [optimize_options, cost_options, common_options]
    ]

    args = _standardize_multiple_optimize_args(
        model,
        samples,
        mapping,
        optimizer,
        *args_options,
    )

    res = _multiple_optimize(model, *args)

    return MultipleOptimize(res)


def _multiple_optimize(
    model: Model,
    samples: Samples,
    mapping: str,
    optimizer: str,
    optimize_options: dict,
    cost_options: dict,
    common_options: dict,
) -> dict:
    if common_options["verbose"]:
        print("</> Multiple Optimize")

    wrap_options = OptionsDT(
        model.setup,
        model.mesh,
        cost_options["njoc"],
        cost_options["njrc"],
    )

    # % Map optimize_options dict to derived type
    _map_dict_to_object(optimize_options, wrap_options.optimize)

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
        shape=(*model.response_data.q.shape, samples.n_sample),
        dtype=np.float32,
        order="F",
    )
    # % Only work with grids (might be changed)
    parameters = np.zeros(
        shape=(
            *model.mesh.flwdir.shape,
            len(optimize_options["parameters"]),
            samples.n_sample,
        ),
        dtype=np.float32,
        order="F",
    )

    wrap_multiple_optimize(
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
        parameters,
    )

    # % Finalize parameters and samples for returns
    parameters = dict(
        zip(
            optimize_options["parameters"],
            np.transpose(parameters, (2, 0, 1, 3)),
        )
    )

    for sp in samples._problem[
        "names"
    ]:  # add uncalibrated parameters from samples to parameters
        if not sp in optimize_options["parameters"]:
            value = getattr(samples, sp)
            value = np.tile(value, (*model.mesh.flwdir.shape, 1))

            parameters.update({sp: value})

    samples_fnl = deepcopy(
        samples
    )  # make a deepcopy of samples (will be modified by setattr)

    for op in optimize_options[
        "parameters"
    ]:  # add calibrated paramters from parameters to samples
        if not op in samples._problem["names"]:
            if op in model.opr_parameters.keys:
                value = model.get_opr_parameters(op)[0, 0]

            elif op in model.opr_initial_states.keys:
                value = model.get_opr_initial_states(op)[0, 0]

            # % In case we have other kind of parameters. Should be unreachable.
            else:
                pass

            setattr(samples_fnl, op, value * np.ones(samples.n_sample))
            setattr(samples_fnl, "_dst_" + op, np.ones(samples.n_sample))

    return {
        "cost": cost,
        "q": q,
        "parameters": parameters,
        "_samples": samples_fnl,
        "_cost_options": cost_options,
    }
