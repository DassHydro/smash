from __future__ import annotations

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

    forward_run(
        instance.setup,
        instance.mesh,
        instance._input_data,
        instance._parameters,
        instance._output,
        options,
        returns,
    )
    # ci, cp, cft, cst, kexc, llr, akw, bkw
    options.optimize.opr_parameters = [0, 1, 1, 0, 1, 0, 1, 1]
    options.optimize.opr_initial_states = [0, 0, 0, 0, 0]
    options.optimize.l_opr_parameters = [
        1e-6,
        1e-6,
        1e-6,
        1e-6,
        -50,
        1e-6,
        1e-3,
        1e-3,
    ]
    options.optimize.u_opr_parameters = [100, 1000, 1000, 10_000, 50, 1000, 50, 1]
    options.optimize.l_opr_initial_states = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
    options.optimize.u_opr_initial_states = [
        0.999999,
        0.999999,
        0.999999,
        0.999999,
        1000,
    ]

    # ~ options.optimize.optimizer = "sbs"
    # ~ options.optimize.mapping = "uniform"
    # ~ options.optimize.maxiter = 5
    # ~ options.optimize.control_tfm = "sbs"

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

    options.optimize.optimizer = "lbfgsb"
    options.optimize.mapping = "distributed"
    options.optimize.maxiter = 10
    options.optimize.control_tfm = "normalize"

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
    # ~ options.optimize.mapping = "multi-linear"
    # ~ options.optimize.maxiter = 100
    # ~ options.optimize.control_tfm = "normalize"
    # ~ opd = np.ones(
    # ~ shape=options.optimize.opr_parameters_descriptor.shape,
    # ~ dtype=np.int32,
    # ~ order="F",
    # ~ )
    # ~ #opd[:, 5] = 0
    # ~ options.optimize.opr_parameters_descriptor = opd
    # ~ options.optimize.opr_initial_states_descriptor = 0

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
