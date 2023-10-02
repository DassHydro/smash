from __future__ import annotations

from smash.core.simulation.optimize._standardize import (
    _standardize_optimize_args,
    _standardize_bayesian_optimize_args,
)
from smash.core.simulation.optimize.optimize import _get_control_info

from copy import deepcopy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model

__all__ = ["optimize_control_info", "bayesian_optimize_control_info"]


def optimize_control_info(
    model: Model,
    mapping: str = "uniform",
    optimizer: str | None = None,
    optimize_options: dict | None = None,
    cost_options: dict | None = None,
) -> dict:
    """
    Get optimize control vector information for the Model object.

    Parameters
    ----------
    model : Model
        Model object.

    mapping : str, default 'uniform'
        Type of mapping. Should be one of 'uniform', 'distributed', 'multi-linear', 'multi-polynomial'

    optimizer : str or None, default None
        Name of optimizer. Should be one of 'sbs', 'lbfgsb'

        .. note::
            If not given, a default optimizer will be set depending on the optimization mapping:

            - **mapping** = 'uniform'; **optimizer** = 'sbs'
            - **mapping** = 'distributed', 'multi-linear', or 'multi-polynomial'; **optimizer** = 'lbfgsb'

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

    Returns
    -------
    control_info : dict
        A dictionary containing optimize control information of the Model object. The elements are:

        - n : int
            The size of the control vector.

        - nbk : np.ndarray
            An array of shape *(4,)* containing the number of elements by kind (opr_parameters, opr_initial_states, serr_mu_parameters, serr_sigma_parameters) of the control vector (``sum(nbk) = n``).

        - x : np.ndarray
            An array of shape *(n,)* containing the initial values of the control vector (it can be transformed).

        - l : np.ndarray
            An array of shape *(n,)* containing the lower bounds of the control vector (it can be transformed).

        - u : np.ndarray
            An array of shape *(n,)* containing the upper bounds of the control vector (it can be transformed).

        - nbd : np.ndarray
            An array of shape *(n,)* containing the type of bounds of the control vector. The values are:

            - 0: unbounded
            - 1: only lower bound
            - 2: both lower and upper bounds
            - 3: only upper bound

        - name : np.ndarray
            An array of shape *(n,)* containing the names of the control vector. The naming convention is:

            - <key>0: Spatially uniform parameter or multi-linear/polynomial intercept where ``<key>`` is the name of any operator parameters or initial_states ('cp0', 'llr0', 'ht0', etc).
            - <key><row>-<col>: Spatially distributed parameter where ``<key>`` is the name of any operator parameters or initial_states and ``<row>``, ``<col>``, the corresponding position in the spatial domain ('cp1-1', 'llr20-2', 'ht3-12', etc). It's one based indexing.
            - <key>-<desc>-<kind>: Multi-linear/polynomial descriptor linked parameter where ``<key>`` is the name of any operator parameters or initial_states, ``<desc>`` the corresponding descriptor and ``<kind>``, the kind of parameter (coefficient or exposant) ('cp-slope-a', 'llr-slope-b', 'ht-dd-a').

        - x_bkg : np.ndarray
            An array of shape *(n,)* containing the background values of the control vector.

        - l_bkg : np.ndarray
            An array of shape *(n,)* containing the background lower bounds of the control vector.

        - u_bkg : np.ndarray
            An array of shape *(n,)* containing the background upper bounds of the control vector.

    Exemples:
    ---------
    TODO: Fill
    """

    args_options = [deepcopy(arg) for arg in [optimize_options, cost_options]]

    # % Only get mapping, optimizer, optimize_options and cost_options
    *args, _, _ = _standardize_optimize_args(
        model,
        mapping,
        optimizer,
        *args_options,
        None,
        None,
    )

    return _get_control_info(model, *args)


def bayesian_optimize_control_info(
    model: Model,
    mapping: str = "uniform",
    optimizer: str | None = None,
    optimize_options: dict | None = None,
    cost_options: dict | None = None,
) -> dict:
    """
    Get bayesian optimize control vector information for the Model object.

    Parameters
    ----------
    model : Model
        Model object.

    mapping : str, default 'uniform'
        Type of mapping. Should be one of 'uniform', 'distributed', 'multi-linear', 'multi-polynomial'.

    optimizer : str or None, default None
        Name of optimizer. Should be one of 'sbs', 'lbfgsb'

        .. note::
            If not given, a default optimizer will be set depending on the optimization mapping:

            - **mapping** = 'uniform'; **optimizer** = 'sbs'
            - **mapping** = 'distributed', 'multi-linear', or 'multi-polynomial'; **optimizer** = 'lbfgsb'

    optimize_options : dict or None, default None
        Dictionary containing optimization options for fine-tuning the optimization process.

        .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element. See the returned parameters in `smash.default_bayesian_optimize_options` for more.

    cost_options : dict or None, default None
        Dictionary containing computation cost options for simulated and observed responses. The elements are:

        gauge : str or ListLike, default 'dws'
            Type of gauge to be computed. There are two ways to specify it:

            - A gauge code or any sequence of gauge codes. The gauge code(s) given must belong to the gauge codes defined in the Model mesh.
            - An alias among 'all' (all gauge codes) and 'dws' (most downstream gauge code(s)).

        control_prior: dict or None, default None
            A dictionary containing the type of prior to link to control parameters. The keys are any control parameter name (i.e. 'cp0', 'cp1-1', 'cp-slope-a', etc), see `smash.bayesian_optimize_control_info` to retrieve control parameters names.
            The values are ListLike of length 2 containing distribution information (i.e. distribution name and parameters). Below, the set of available distributions and the associated number of parameters:

            - 'FlatPrior', [] (0)
            - 'Gaussian', [mu, sigma] (2)
            - 'Exponential', [threshold, scale] (2)

            .. note:: If not given, a 'FlatPrior' is set to each control parameters (equivalent to no prior)

        end_warmup : str or pandas.Timestamp, default model.setup.start_time
            The end of the warm-up period, which must be between the start time and the end time defined in the Model setup. By default, it is set to be equal to the start time.

        .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

    Returns
    -------
    control_info : dict
        A dictionary containing optimize control information of the Model object. The elements are:

        - n : int
            The size of the control vector.

        - nbk : np.ndarray
            An array of shape *(4,)* containing the number of elements by kind (opr_parameters, opr_initial_states, serr_mu_parameters, serr_sigma_parameters) of the control vector (``sum(nbk) = n``).

        - x : np.ndarray
            An array of shape *(n,)* containing the initial values of the control vector (it can be transformed).

        - l : np.ndarray
            An array of shape *(n,)* containing the lower bounds of the control vector (it can be transformed).

        - u : np.ndarray
            An array of shape *(n,)* containing the upper bounds of the control vector (it can be transformed).

        - nbd : np.ndarray
            An array of shape *(n,)* containing the type of bounds of the control vector. The values are:

            - 0: unbounded
            - 1: only lower bound
            - 2: both lower and upper bounds
            - 3: only upper bound

        - name : np.ndarray
            An array of shape *(n,)* containing the names of the control vector. The naming convention is:

            - <key>0: Spatially uniform parameter or multi-linear/polynomial intercept where ``<key>`` is the name of any operator parameters or initial_states ('cp0', 'llr0', 'ht0', etc).
            - <key><row>-<col>: Spatially distributed parameter where ``<key>`` is the name of any operator parameters or initial_states and ``<row>``, ``<col>``, the corresponding position in the spatial domain ('cp1-1', 'llr20-2', 'ht3-12', etc). It's one based indexing.
            - <key>-<desc>-<kind>: Multi-linear/polynomial descriptor linked parameter where ``<key>`` is the name of any operator parameters or initial_states, ``<desc>`` the corresponding descriptor and ``<kind>``, the kind of parameter (coefficient or exposant) ('cp-slope-a', 'llr-slope-b', 'ht-dd-a').
            - <key>-<code>: Structural error parameter where ``<key>`` is the name of any structural error mu or sigma parameters and ``<code>`` the corresponding gauge ('sg0-V3524010', 'sg1-V3524010', etc)

        - x_bkg : np.ndarray
            An array of shape *(n,)* containing the background values of the control vector.

        - l_bkg : np.ndarray
            An array of shape *(n,)* containing the background lower bounds of the control vector.

        - u_bkg : np.ndarray
            An array of shape *(n,)* containing the background upper bounds of the control vector.

    Exemples:
    ---------
    TODO: Fill
    """

    args_options = [deepcopy(arg) for arg in [optimize_options, cost_options]]

    # % Only get mapping, optimizer, optimize_options and cost_options
    *args, _, _ = _standardize_bayesian_optimize_args(
        model,
        mapping,
        optimizer,
        *args_options,
        None,
        None,
    )

    return _get_control_info(model, *args)
