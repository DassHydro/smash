from __future__ import annotations

from smash._constant import (
    DEFAULT_BOUNDS_OPR_PARAMETERS,
    DEFAULT_BOUNDS_OPR_INITIAL_STATES,
    OPTIMIZABLE_OPR_PARAMETERS,
    OPTIMIZABLE_OPR_INITIAL_STATES,
)

from smash.core.simulation._standardize import (
    _standardize_options,
    _standardize_returns,
)

from smash.fcore._mw_forward import forward_run
from smash.fcore._mw_optimize import sbs_optimize, lbfgsb_optimize

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.fcore._mwd_options import OptionsDT
    from smash.fcore._mwd_returns import ReturnsDT

__all__ = ["optimize"]


def optimize(
    model: Model, options: OptionsDT | None = None, returns: ReturnsDT | None = None
):
    new_model = model.copy()

    _optimize(new_model, options, returns)

    return new_model


def _optimize(instance: Model, options: OptionsDT, returns: ReturnsDT):
    options = _standardize_options(options, instance.setup)

    returns = _standardize_returns(returns)

    options.comm.ncpu = 6

    # ~ forward_run(
    # ~ instance.setup,
    # ~ instance.mesh,
    # ~ instance._input_data,
    # ~ instance._parameters,
    # ~ instance._output,
    # ~ options,
    # ~ returns,
    # ~ )

    for i, key in enumerate(instance.opr_parameters.keys):
        options.optimize.opr_parameters[i] = OPTIMIZABLE_OPR_PARAMETERS[key]
        options.optimize.l_opr_parameters[i] = DEFAULT_BOUNDS_OPR_PARAMETERS[key][0]
        options.optimize.u_opr_parameters[i] = DEFAULT_BOUNDS_OPR_PARAMETERS[key][1]

    options.optimize.opr_parameters_descriptor = 1

    for i, key in enumerate(instance.opr_initial_states.keys):
        options.optimize.opr_initial_states[i] = 0
        options.optimize.l_opr_initial_states[i] = DEFAULT_BOUNDS_OPR_INITIAL_STATES[
            key
        ][0]
        options.optimize.u_opr_initial_states[i] = DEFAULT_BOUNDS_OPR_INITIAL_STATES[
            key
        ][1]

    options.optimize.opr_initial_states_descriptor = 1

    options.optimize.optimizer = "sbs"
    options.optimize.mapping = "uniform"
    options.optimize.maxiter = 10
    options.optimize.control_tfm = "sbs"

    optimize_func = eval(options.optimize.optimizer + "_optimize")

    optimize_func(
        instance.setup,
        instance.mesh,
        instance._input_data,
        instance._parameters,
        instance._output,
        options,
        returns,
    )

    # ~ options.optimize.optimizer = "lbfgsb"
    # ~ options.optimize.mapping = "distributed"
    # ~ options.optimize.maxiter = 100
    # ~ options.optimize.control_tfm = "normalize"

    # ~ optimize_func = eval(options.optimize.optimizer + "_optimize")

    # ~ optimize_func(
    # ~ instance.setup,
    # ~ instance.mesh,
    # ~ instance._input_data,
    # ~ instance._parameters,
    # ~ instance._output,
    # ~ options,
    # ~ returns,
    # ~ )

    # ~ options.optimize.optimizer = "lbfgsb"
    # ~ options.optimize.mapping = "multi-polynomial"
    # ~ options.optimize.maxiter = 100
    # ~ options.optimize.control_tfm = "normalize"

    # ~ optimize_func = eval(options.optimize.optimizer + "_optimize")

    # ~ optimize_func(
    # ~ instance.setup,
    # ~ instance.mesh,
    # ~ instance._input_data,
    # ~ instance._parameters,
    # ~ instance._output,
    # ~ options,
    # ~ returns,
    # ~ )
