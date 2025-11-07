from __future__ import annotations

from textwrap import dedent

from smash._constant import (
    DEFAULT_SIMULATION_COMMON_OPTIONS,
    DEFAULT_SIMULATION_COST_OPTIONS,
    DEFAULT_SIMULATION_RETURN_OPTIONS,
)
from smash.util._doctools import DocAppender, DocSubstitution

# TODO: store the docstring for each returned variable and then applied it
# for each optional return object (ForwardRun, Optimize, etc.)

OPTIMIZE_OPTIONS_KEYS_DOC = [
    "parameters",
    "bounds",
    "control_tfm",
    "descriptor",
    "net",
    "learning_rate",
    "random_state",
    "termination_crit",
]

BAYESIAN_OPTIMIZE_OPTIONS_KEYS_DOC = [
    "parameters",
    "bounds",
    "control_tfm",
    "descriptor",
    "termination_crit",
]

RETURN_CONTROL_INFO_BASE_DOC = """
Returns
-------
control_info : `dict[str, Any]`
    A dictionary containing optimize control information of Model. The elements are:

    - n : `int`
        The size of the control vector.

    - nbk : `numpy.ndarray`
        An array of shape *(6,)* containing the number of elements by kind (`Model.rr_parameters`,
        `Model.rr_initial_states`, `Model.serr_mu_parameters`, `Model.serr_sigma_parameters`,
        `Model.nn_parameters`, `Net <factory.Net>`) of the control vector (``sum(nbk) = n``).

    - x : `numpy.ndarray`
        An array of shape *(n,)* containing the initial values of the control vector (it can be transformed).

    - l : `numpy.ndarray`
        An array of shape *(n,)* containing the lower bounds of the control vector (it can be transformed).

    - u : `numpy.ndarray`
        An array of shape *(n,)* containing the upper bounds of the control vector (it can be transformed).

    - nbd : `numpy.ndarray`
        An array of shape *(n,)* containing the type of bounds of the control vector. The values are:

        - ``0``: unbounded
        - ``1``: only lower bound
        - ``2``: both lower and upper bounds
        - ``3``: only upper bound

    - x_raw : `numpy.ndarray`
        An array of shape *(n,)* containing the raw (non-transformed) values of the control vector.

    - l_raw : `numpy.ndarray`
        An array of shape *(n,)* containing the raw (non-transformed) lower bounds of the control vector.

    - u_raw : `numpy.ndarray`
        An array of shape *(n,)* containing the raw (non-transformed) upper bounds of the control vector.
"""

MAPPING_OPTIMIZER_BASE_DOC = {
    "mapping": (
        """
        `str`, default 'uniform'
        """,
        """
        Type of mapping. Should be one of

        - ``'uniform'``
        - ``'distributed'``
        - ``'multi-linear'``
        - ``'multi-power'``
        %(mapping_ann)s

        .. hint::
            See the :ref:`math_num_documentation.mapping` section.
        """,
    ),
    "optimizer": (
        """
        `str` or None, default None
        """,
        """
        Name of optimizer. Should be one of

        - ``'sbs'`` (only for ``'uniform'`` **mapping**)
        - ``'nelder-mead'`` (only for ``'uniform'`` **mapping**)
        - ``'powell'`` (only for ``'uniform'`` **mapping**)
        %(optimizer_lbfgsb)s
        - ``'adam'`` (for all mappings)
        - ``'adagrad'`` (for all mappings)
        - ``'rmsprop'`` (for all mappings)
        - ``'sgd'`` (for all mappings)

        .. note::
            If not given, a default optimizer will be set as follows:

            - ``'sbs'`` for **mapping** = ``'uniform'``
            - ``'lbfgsb'`` for **mapping** = ``'distributed'``, ``'multi-linear'``, ``'multi-power'``
            %(default_optimizer_for_ann_mapping)s

        .. hint::
            See the :ref:`math_num_documentation.optimization_algorithms` section.
        """,
    ),
}

OPTIMIZE_OPTIONS_BASE_DOC = {
    "parameters": (
        """
        `str`, `list[str, ...]` or None, default None
        """,
        """
        Name of parameters to optimize. Should be one or a sequence of any key of:

        - `Model.rr_parameters`
        - `Model.rr_initial_states`
        - `Model.nn_parameters`, if using a hybrid model structure (depending on **hydrological_module**)
        %(parameters_serr_mu_parameters)s
        %(parameters_serr_sigma_parameters)s

        >>> optimize_options = {
            "parameters": "cp",
        }
        >>> optimize_options = {
            "parameters": ["cp", "ct", "kexc", "llr"],
        }

        .. note::
            If not given, all parameters in `Model.rr_parameters`, `Model.nn_parameters` (if used)
            %(parameters_note_serr_parameters)s will be optimized.
        """,
    ),
    "bounds": (
        """
        `dict[str, tuple[float, float]]` or None, default None
        """,
        """
        Bounds on optimized parameters. A dictionary where the keys represent parameter names, and the values
        are pairs of ``(min, max)`` values (i.e., a list or tuple) with ``min`` lower than ``max``. The keys
        must be included in **parameters**.

        >>> optimize_options = {
            "bounds": {
                "cp": (1, 2000),
                "ct": (1, 1000),
                "kexc": (-10, 5)
                "llr": (1, 1000)
            },
        }

        .. note::
            If not given, default bounds will be applied to each parameter.
            See `Model.get_rr_parameters_bounds`,
            `Model.get_rr_initial_states_bounds`%(bounds_get_serr_parameters_bounds)s
        """,
    ),
    "control_tfm": (
        """
        `str` or None, default None
        """,
        """
        Transformation method applied to bounded parameters of the control vector. Should be one of

        - ``'keep'``
        - ``'normalize'``
        - ``'sbs'`` (``'sbs'`` **optimizer** only)

        .. note::
            If not given, the default control vector transformation is **control_tfm** = ``'normalize'``
            except for the ``'sbs'`` optimizer, where **control_tfm** = ``'sbs'``. This options is not used
            when **mapping** is ``'ann'``.
        """,
    ),
    "descriptor": (
        """
        `dict[str, list[str, ...]]` or None, default None
        """,
        """
        Descriptors linked to optimized parameters. A dictionary where the keys represent parameter names, and
        the values are list of descriptor names. The keys must be included in **parameters**.

        >>> optimize_options = {
            "descriptor": {
                "cp": ["slope", "dd"],
                "ct": ["slope"],
                "kexc": ["slope", "dd"],
                "llr": ["dd"],
            },
        }

        .. note::
            If not given, all descriptors will be used for each parameter.
            This option is only be used when **mapping** is ``'multi-linear'`` or ``'multi-power'``.
            In case of ``'ann'``, all descriptors will be used.
        """,
    ),
    "net": (
        """
        `Net <factory.Net>` or None, default None
        """,
        """
        The regionalization neural network used to learn the descriptors-to-parameters mapping.

        .. note::
            If not given, a default neural network will be used. This option is only used when **mapping** is
            ``'ann'``. See `Net <factory.Net>` to learn how to create a customized neural network for
            training.
        """,
    ),
    "learning_rate": (
        """
        `float` or None, default None
        """,
        """
        The learning rate used for updating trainable parameters when using adaptive optimizers
        (i.e., ``'adam'``, ``'adagrad'``, ``'rmsprop'``, ``'sgd'``).

        .. note::
            If not given, a default learning rate for each optimizer will be used.
        """,
    ),
    "random_state": (
        """
        `int` or None, default None
        """,
        """
        A random seed used to initialize neural network parameters.

        .. note::
            If not given, the neural network parameters will be initialized with a random seed. This options
            is only used when **mapping** is ``'ann'``, and the weights and biases of **net** are not yet
            initialized.
        """,
    ),
    "termination_crit": (
        """
        `dict[str, Any]` or None, default None
        """,
        """
        Termination criteria. The elements are:

        - ``'maxiter'``: The maximum number of iterations.
        - ``'xatol'``: Absolute error in solution parameters between iterations that is acceptable for
          convergence. Only used when **optimizer** is ``'nelder-mead'``.
        - ``'fatol'``: Absolute error in cost function value between iterations that is acceptable for
          convergence. Only used when **optimizer** is ``'nelder-mead'``.
        - ``'factr'``: An additional termination criterion based on cost values. Only used when **optimizer**
          is ``'lbfgsb'``.
        - ``'pgtol'``: An additional termination criterion based on the projected gradient of the cost
          function. Only used when **optimizer** is ``'lbfgsb'``.
        - ``'early_stopping'``: A positive number to stop training when the cost function does not decrease
          below the current optimal value for **early_stopping** consecutive iterations. When set to zero,
          early stopping is disabled, and the training continues for the full number of iterations.
          Only used for adaptive optimizers (i.e., ``'adam'``, ``'adagrad'``, ``'rmsprop'``, ``'sgd'``).

        >>> optimize_options = {
            "termination_crit": {
                "maxiter": 10,
                "factr": 1e6,
            },
        }
        >>> optimize_options = {
            "termination_crit": {
                "maxiter": 200,
                "early_stopping": 20,
            },
        }

        .. note::
            If not given, default values are set to each elements.
        """,
    ),
}

COST_OPTIONS_BASE_DOC = {
    "jobs_cmpt": (
        """
        `str` or `list[str, ...]`, default 'nse'
        """,
        """
        Type of observation objective function(s) to be computed. Should be one or a sequence of any of

        - ``'nse'``, ``'nnse'``, ``'kge'``, ``'mae'``, ``'mape'``, ``'mse'``, ``'rmse'``, ``'lgrm'``
          (classical evaluation metrics)
        - ``'Crc'``, ``'Crchf'``, ``'Crclf'``, ``'Crch2r'``, ``'Cfp2'``, ``'Cfp10'``, ``'Cfp50'``, ``'Cfp90'``
          (continuous signatures-based error metrics)
        - ``'Eff'``, ``'Ebf'``, ``'Erc'``, ``'Erchf'``, ``'Erclf'``, ``'Erch2r'``, ``'Elt'``, ``'Epf'``
          (flood event signatures-based error metrics)

        >>> cost_options = {
            "jobs_cmpt": "nse",
        }
        >>> cost_options = {
            "jobs_cmpt": ["nse", "Epf"],
        }

        .. hint::
            See the :ref:`math_num_documentation.efficiency_error_metric` and
            :ref:`math_num_documentation.hydrological_signature` sections
        """,
    ),
    "jobs_cmpt_tfm": (
        """
        `str` or `list[str, ...]`, default 'keep'
        """,
        """
        Type of transformation applied to discharge in observation objective function(s). Should be one or a
        sequence of any of

        - ``'keep'`` : No transformation :math:`f:x \\rightarrow x`
        - ``'sqrt'`` : Square root transformation :math:`f:x \\rightarrow \\sqrt{x}`
        - ``'inv'`` : Multiplicative inverse transformation :math:`f:x \\rightarrow \\frac{1}{x}`

        >>> cost_options = {
            "jobs_cmpt_tfm": "inv",
        }
        >>> cost_options = {
            "jobs_cmpt_tfm": ["keep", "inv"],
        }

        .. note::
            If **jobs_cmpt** is a list of multi-objective functions, and only one transformation is chosen in
            **jobs_cmpt_tfm**, the transformation will be applied to each observation objective function.
        """,
    ),
    "wjobs_cmpt": (
        """
        `str` or `list[float, ...]`, default 'mean'
        """,
        """
        The corresponding weighting of observation objective functions in case of multi-criteria
        (i.e., a sequence of objective functions to compute). There are two ways to specify it:

        - An alias among ``'mean'``
        - A sequence of value whose size must be equal to the number of observation objective function(s) in
          **jobs_cmpt**

        >>> cost_options = {
            "wjobs_cmpt": "mean",
        }
        >>> cost_options = {
            "wjobs_cmpt": [0.7, 0.3],
        }
        """,
    ),
    "wjreg": (
        """
        `float` or `str`, default 0
        """,
        """
        The weighting of regularization term. There are two ways to specify it:

        - A value greater than or equal to 0
        - An alias among ``'fast'`` or ``'lcurve'``. **wjreg** will be auto-computed by one of these methods.

        >>> cost_options = {
            "wjreg": 1e-4,
        }
        >>> cost_options = {
            "wjreg": "lcurve",
        }

        .. hint::
            See the :ref:`math_num_documentation.cost_function.regularization_weighting_coefficient` section
        """,
    ),
    "jreg_cmpt": (
        """
        `str` or `list[str, ...]`, default 'prior'
        """,
        """
        Type(s) of regularization function(s) to be minimized when regularization term is set
        (i.e., **wjreg** > 0). Should be one or a sequence of any of

        - ``'prior'``
        - ``'smoothing'``
        - ``'hard-smoothing'``

        >>> cost_options = {
            "jreg_cmpt": "prior",
        }
        >>> cost_options = {
            "jreg_cmpt": ["prior", "smoothing"],
        }

        .. hint::
            See the :ref:`math_num_documentation.regularization_function` section
        """,
    ),
    "wjreg_cmpt": (
        """
        `str` or `list[float, ...]`, default 'mean'
        """,
        """
        The corresponding weighting of regularization functions in case of multi-regularization
        (i.e., a sequence of regularization functions to compute). There are two ways to specify it:

        - An alias among ``'mean'``
        - A sequence of value whose size must be equal to the number of regularization function(s) in
          **jreg_cmpt**

        >>> cost_options = {
            "wjreg_cmpt": "mean",
        }
        >>> cost_options = {
            "wjreg_cmpt": [1., 2.],
        }
        """,
    ),
    "end_warmup": (
        """
        `str`, `pandas.Timestamp` or None, default None
        """,
        """
        The end of the warm-up period, which must be between the start time and the end time defined in
        `Model.setup`.

        >>> cost_options = {
            "end_warmup": "1997-12-21",
        }
        >>> cost_options = {
            "end_warmup": pd.Timestamp("19971221"),
        }

        .. note::
            If not given, it is set to be equal to the `Model.setup` start time.
        """,
    ),
    "gauge": (
        """
        `str` or `list[str, ...]`, default 'dws'
        """,
        """
        Type of gauge to be computed. There are two ways to specify it:

        - An alias among ``'all'`` (all gauge codes) or ``'dws'`` (most downstream gauge code(s))
        - A gauge code or any sequence of gauge codes. The gauge code(s) given must belong to the gauge codes
          defined in the `Model.mesh`

        >>> cost_options = {
            "gauge": "dws",
        }
        >>> cost_options = {
            "gauge": "V3524010",
        }
        >>> cost_options = {
            "gauge": ["V3524010", "V3515010"],
        }
        """,
    ),
    "wgauge": (
        """
        `str` or `list[float, ...]` default 'mean'
        """,
        """
        Type of gauge weights. There are two ways to specify it:

        - An alias among ``'mean'``, ``'lquartile'`` (1st quantile or lower quantile), ``'median'``, or
          ``'uquartile'`` (3rd quantile or upper quantile)
        - A sequence of value whose size must be equal to the number of gauges optimized in **gauge**

        >>> cost_options = {
            "wgauge": "mean",
        }
        >>> cost_options = {
            "wgauge": [0.6, 0.4]",
        }
        """,
    ),
    "control_prior": (
        """
        `dict[str, list[str, list[float]]]` or None, default None
        """,
        """
        Prior applied to the control vector.
        A dictionary containing the type of prior to link to control vector. The keys are any control
        parameter name (i.e. ``'cp-0'``, ``'cp-1-1'``, ``'cp-slope-a'``, etc.), see
        `bayesian_optimize_control_info` to retrieve control parameters
        names. The values are list of length 2 containing distribution information (i.e. distribution name and
        parameters). Below, the set of available distributions and the associated number of parameters:

        - ``'FlatPrior'``,   []                                 (0)
        - ``'Uniform'``,     [lower_bound, higher_bound]        (2)
        - ``'Gaussian'``,    [mean, standard_deviation]         (2)
        - ``'Exponential'``, [threshold, scale]                 (2)
        - ``'LogNormal'``,   [mean_log, standard_deviation_log] (2)
        - ``'Triangle'``,    [peak, lower_bound, higher_bound]  (3)

        >>> cost_options = {
            "control_prior": {
                "cp-0": ["Gaussian", [200, 100]],
                "kexc-0": ["Gaussian", [0, 5]],
            }
        }

        .. note::
            If not given, ``'FlatPrior'`` is applied to each control vector parameter (i.e. equivalent to no
            prior).

        .. hint::
            See a more detailed explanation on the available distributions in
            :ref:`math_num_documentation.bayesian_estimation` section.
        """,
    ),
    "event_seg": (
        """
        `dict[str, float]`, default {'peak_quant': 0.995, 'max_duration': 240}
        """,
        """
        A dictionary of event segmentation options when calculating flood event signatures for cost
        computation (i.e., **jobs_cmpt** includes flood events signatures).

        >>> cost_options = {
            "event_seg": {
                "peak_quant": 0.998,
                "max_duration": 120,
            }
        }

        .. hint::
            See the `hydrograph_segmentation` function and
            :ref:`math_num_documentation.hydrograph_segmentation` section.

        """,
    ),
}

COMMON_OPTIONS_BASE_DOC = {
    "verbose": (
        """
        `bool`, default False
        """,
        """
        Whether to display information about the running method.
        """,
    ),
    "ncpu": (
        """
        `int`, default 1
        """,
        """
        Number of CPU(s) to perform a parallel computation.

        .. warning::
            Parallel computation is not supported on ``Windows``.
        """,
    ),
}

RETURN_OPTIONS_BASE_DOC = {
    "time_step": (
        """
        `str`, `pandas.Timestamp`, `pandas.DatetimeIndex` or `list[str, ...]`, default 'all'
        """,
        """
        Returned time steps. There are five ways to specify it:

        - An alias among ``'all'`` (return all time steps).
        - A date as string which respect `pandas.Timestamp` format
        - A `pandas.Timestamp`.
        - A `pandas.DatetimeIndex`.
        - A sequence of dates as strings.

        >>> return_options = {
            "time_step": "all",
        }
        >>> return_options = {
            "time_step": "1997-12-21",
        }
        >>> return_options = {
            "time_step": pd.Timestamp("19971221"),
        }
        >>> return_options = {
            "time_step": pd.date_range(
                start="1997-12-21",
                end="1998-12-21",
                freq="1D"
            ),
        }
        >>> return_options = {
            "time_step": ["1998-05-23", "1998-05-24", "1998-05-25"],
        }

        .. note::
            It only applies to the following variables: ``'rr_states'`` and ``'q_domain'``
        """,
    ),
    "rr_states": (
        """
        `bool`, default False
        """,
        """
        Whether to return rainfall-runoff states for specific time steps.
        """,
    ),
    "q_domain": (
        """
        `bool`, default False
        """,
        """
        Whether to return simulated discharge on the whole domain for specific time steps.
        """,
    ),
    "internal_fluxes": (
        """
        `bool`, default False
        """,
        """
        Whether to return internal fluxes depending on the model structure on \
        the whole domain for specific time steps.
        """,
    ),
    "control_vector": (
        """
        `bool`, default False
        """,
        """
        Whether to return the control vector solution of the optimization (it can be transformed).
        """,
    ),
    "net": (
        """
        `bool`, default False
        """,
        """
        Whether to return the trained neural network `Net <factory.Net>`. Only used with ``'ann'``
        **mapping**.
        """,
    ),
    "cost": (
        """
        `bool`, default False
        """,
        """
        Whether to return cost value.
        """,
    ),
    "n_iter": (
        """
        `bool`, default False
        """,
        """
        Whether to return the number of iterations performed.
        """,
    ),
    "projg": (
        """
        `bool`, default False
        """,
        """
        Whether to return the projected gradient value (infinity norm of the Jacobian matrix).
        """,
    ),
    "jobs": (
        """
        `bool`, default False
        """,
        """
        Whether to return jobs (observation component of cost) value.
        """,
    ),
    "jreg": (
        """
        `bool`, default False
        """,
        """
        Whether to return jreg (regularization component of cost) value.
        """,
    ),
    "lcurve_wjreg": (
        """
        `bool`, default False
        """,
        """
        Whether to return the wjreg L-curve. Only used if **wjreg** in cost_options is equal to ``'lcurve'``.
        """,
    ),
    "lcurve_multiset": (
        """
        `bool`, default False
        """,
        """
        Whether to return the multiset estimate L-curve.
        """,
    ),
    "log_lkh": (
        """
        `bool`, default False
        """,
        """
        Whether to return log likelihood component value.
        """,
    ),
    "log_prior": (
        """
        `bool`, default False
        """,
        """
        Whether to return log prior component value.
        """,
    ),
    "log_h": (
        """
        `bool`, default False
        """,
        """
        Whether to return log h component value.
        """,
    ),
    "serr_mu": (
        """
        `bool`, default False
        """,
        """
        Whether to return mu, the mean of structural errors. It can also be returned directly from the Model
        object using the `Model.get_serr_mu` method.
        """,
    ),
    "serr_sigma": (
        """
        `bool`, default False
        """,
        """
        Whether to return sigma, the standard deviation of structural errors. It can also be returned directly
        from the Model object using the `Model.get_serr_sigma` method.
        """,
    ),
}


def _gen_docstring_from_base_doc(
    base_doc: dict[str, tuple[str, str]], keys: list[str], nindent: int = 0
) -> str:
    ret = []
    for key in keys:
        typ = dedent(base_doc[key][0].strip())
        description = dedent(base_doc[key][1].strip())
        text = f"{key} : {typ}\n\t{description}\n"

        ret.append(nindent * "    " + text)
    return "\n".join(ret)


_forward_run_doc = (
    # % TODO FC: Add advanced user guide
    """
Run the forward Model.

Parameters
----------
%(model_parameter)s
cost_options : `dict[str, Any]` or None, default None
    Dictionary containing computation cost options for simulated and observed responses. The elements are:

"""
    + _gen_docstring_from_base_doc(
        COST_OPTIONS_BASE_DOC,
        DEFAULT_SIMULATION_COST_OPTIONS["forward_run"].keys(),
        nindent=1,
    )
    + """
common_options : `dict[str, Any]` or None, default None
    Dictionary containing common options with two elements:

"""
    + _gen_docstring_from_base_doc(
        COMMON_OPTIONS_BASE_DOC, DEFAULT_SIMULATION_COMMON_OPTIONS.keys(), nindent=1
    )
    + """
return_options : `dict[str, Any]` or None, default None
    Dictionary containing return options to save additional simulation results. The elements are:

"""
    + _gen_docstring_from_base_doc(
        RETURN_OPTIONS_BASE_DOC,
        DEFAULT_SIMULATION_RETURN_OPTIONS["forward_run"].keys(),
        nindent=1,
    )
    + """
Returns
-------
%(model_return)s
forward_run : `ForwardRun` or None, default None
    It returns an object containing additional simulation results with the keys defined in
    **return_options**. If no keys are defined, it returns None.

See Also
--------
ForwardRun : Represents forward run optional results.

Examples
--------
>>> from smash.factory import load_dataset
>>> setup, mesh = load_dataset("cance")
>>> model = smash.Model(setup, mesh)

Run the direct Model

>>> %(model_example_func)s
</> Forward Run

Get the simulated discharges

>>> %(model_example_response)s.response.q
array([[1.9826430e-03, 1.3466669e-07, 6.7617895e-12, ..., 2.2796249e+01,
        2.2655941e+01, 2.2517307e+01],
       [2.3777038e-04, 7.3761623e-09, 1.7551447e-13, ..., 4.8298149e+00,
        4.8079352e+00, 4.7862868e+00],
       [2.9721676e-05, 5.4272520e-10, 8.4623445e-15, ..., 1.2818875e+00,
        1.2760198e+00, 1.2702127e+00]], dtype=float32)
"""
)

_optimize_doc = (
    # % TODO FC: Add advanced user guide
    """
Model assimilation using numerical optimization algorithms.

Parameters
----------
%(model_parameter)s

"""
    + _gen_docstring_from_base_doc(MAPPING_OPTIMIZER_BASE_DOC, ["mapping", "optimizer"], nindent=0)
    + """

optimize_options : `dict[str, Any]` or None, default None
    Dictionary containing optimization options for fine-tuning the optimization process.
    See `%(default_optimize_options_func)s` to retrieve the default optimize options based on the **mapping**
    and **optimizer**.

"""
    + _gen_docstring_from_base_doc(
        OPTIMIZE_OPTIONS_BASE_DOC,
        OPTIMIZE_OPTIONS_KEYS_DOC,
        nindent=1,
    )
    + """
cost_options : `dict[str, Any]` or None, default None
    Dictionary containing computation cost options for simulated and observed responses. The elements are:

"""
    + _gen_docstring_from_base_doc(
        COST_OPTIONS_BASE_DOC,
        DEFAULT_SIMULATION_COST_OPTIONS["optimize"].keys(),
        nindent=1,
    )
    + """
common_options : `dict[str, Any]` or None, default None
    Dictionary containing common options with two elements:

"""
    + _gen_docstring_from_base_doc(
        COMMON_OPTIONS_BASE_DOC, DEFAULT_SIMULATION_COMMON_OPTIONS.keys(), nindent=1
    )
    + """
return_options : `dict[str, Any]` or None, default None
    Dictionary containing return options to save additional simulation results. The elements are:

"""
    + _gen_docstring_from_base_doc(
        RETURN_OPTIONS_BASE_DOC,
        DEFAULT_SIMULATION_RETURN_OPTIONS["optimize"].keys(),
        nindent=1,
    )
    + """
callback : callable or None, default None
    A callable called after each iteration with the signature ``callback(iopt: Optimize)``, where
    ``iopt`` is a keyword argument representing an instance of the `Optimize` class that contains
    intermediate optimization results with attributes:

    - ``'control_vector'``: The current control vector.
    - ``'cost'``: The current cost value.
    - ``'n_iter'``: The current number of iterations performed by the optimizer.
    - ``'projg'``: The current projected gradient, available if using gradient-based optimizers.
    - ``'net'``: The regionalization neural network state, available if using ``'ann'`` **mapping**.

    >>> import numpy as np
    >>> iter_cost = []  # to get the cost values through iterations
    >>> def callback_func(iopt, icost=iter_cost):
    ...     icost.append(iopt.cost)
    ...     # save the current control vector value to a text file
    ...     np.savetxt(f"control_iter_{len(icost)}.txt", iopt.control_vector)
    >>> callback = callback_func

    .. note::
        The name of the argument must be ``iopt`` for the callback to be passed as an `Optimize` object.

"""
    + """
Returns
-------
%(model_return)s
optimize : `Optimize` or None, default None
    It returns an object containing additional simulation results with the keys defined in
    **return_options**. If no keys are defined, it returns None.

See Also
--------
Optimize : Represents optimize optional results.

Examples
--------
>>> from smash.factory import load_dataset
>>> setup, mesh = load_dataset("cance")
>>> model = smash.Model(setup, mesh)

Optimize the Model

>>> %(model_example_func)s
</> Optimize
    At iterate     0    nfg =     1    J = 6.95010e-01    ddx = 0.64
    At iterate     1    nfg =    30    J = 9.84107e-02    ddx = 0.64
    At iterate     2    nfg =    59    J = 4.54087e-02    ddx = 0.32
    At iterate     3    nfg =    88    J = 3.81818e-02    ddx = 0.16
    At iterate     4    nfg =   117    J = 3.73617e-02    ddx = 0.08
    At iterate     5    nfg =   150    J = 3.70873e-02    ddx = 0.02
    At iterate     6    nfg =   183    J = 3.68004e-02    ddx = 0.02
    At iterate     7    nfg =   216    J = 3.67635e-02    ddx = 0.01
    At iterate     8    nfg =   240    J = 3.67277e-02    ddx = 0.01
    CONVERGENCE: DDX < 0.01

Get the simulated discharges

>>> %(model_example_response)s.response.q
array([[5.8217300e-04, 4.7552472e-04, 3.5390016e-04, ..., 1.9405001e+01,
        1.9179874e+01, 1.8959581e+01],
       [1.2144940e-04, 6.6219603e-05, 3.0706153e-05, ..., 4.7972722e+00,
        4.7477250e+00, 4.6991367e+00],
       [1.9631812e-05, 6.9778694e-06, 2.2202112e-06, ..., 1.2500964e+00,
        1.2371680e+00, 1.2244837e+00]], dtype=float32)
"""
)

_multiset_estimate_doc = (
    # % TODO FC: Add advanced user guide
    """
Model assimilation using Bayesian-like estimation on multiple sets of solutions.

Parameters
----------
%(model_parameter)s

multiset : `MultipleForwardRun <MultipleForwardRun>`
    The returned object created by `multiple_forward_run` method containing
    information about multiple sets of rainfall-runoff parameters or initial states.

alpha : `float`, `list[float, ...]`, or None, default None
    A regularization parameter that controls the decay rate of the likelihood function.
    If **alpha** is a list, the L-curve approach will be used to find an optimal value for the regularization
    parameter.

    .. note::
        If not given, a default numeric range will be set for optimization through the L-curve process.

common_options : `dict[str, Any]` or None, default None
    Dictionary containing common options with two elements:

"""
    + _gen_docstring_from_base_doc(
        COMMON_OPTIONS_BASE_DOC,
        DEFAULT_SIMULATION_COMMON_OPTIONS.keys(),
        nindent=1,
    )
    + """
return_options : `dict[str, Any]` or None, default None
    Dictionary containing return options to save additional simulation results. The elements are:

"""
    + _gen_docstring_from_base_doc(
        RETURN_OPTIONS_BASE_DOC,
        DEFAULT_SIMULATION_RETURN_OPTIONS["multiset_estimate"].keys(),
        nindent=1,
    )
    + """
Returns
-------
%(model_return)s

multiset_estimate : `MultisetEstimate` or None, default None
    It returns an object containing additional simulation results with the keys defined in
    **return_options**. If no keys are defined, it returns None.

See Also
--------
MultisetEstimate : Represents multiset estimate optional results.
MultipleForwardRun : Represents multiple forward run computation result.

Examples
--------
>>> from smash.factory import load_dataset
>>> from smash.factory import generate_samples
>>> setup, mesh = load_dataset("cance")
>>> model = smash.Model(setup, mesh)

Define sampling problem and generate samples

>>> problem = {
...    'num_vars': 4,
...    'names': ['cp', 'ct', 'kexc', 'llr'],
...    'bounds': [[1, 2000], [1, 1000], [-20, 5], [1, 1000]]
... }
>>> sr = generate_samples(problem, n=100, random_state=11)

Run Model with multiple sets of parameters

>>> mfr = smash.multiple_forward_run(model, samples=sr)
</> Multiple Forward Run
    Forward Run 100/100 (100%(percent)s)

Estimate Model on multiple sets of solutions

>>> %(model_example_func)s
</> Multiple Set Estimate
    L-curve Computing: 100%(percent)s|████████████████████████████████████████| 50/50 [00:02<00:00, 17.75it/s]

Get the simulated discharges

>>> %(model_example_response)s.response.q
array([[9.4571486e-05, 9.3920688e-05, 9.3143637e-05, ..., 1.7423288e+01,
        1.7193638e+01, 1.6963835e+01],
       [2.5758292e-05, 2.4744744e-05, 2.3561088e-05, ..., 3.6616585e+00,
        3.6165960e+00, 3.5724759e+00],
       [5.9654208e-06, 5.0872231e-06, 4.2139386e-06, ..., 9.0600485e-01,
        8.9525825e-01, 8.8473159e-01]], dtype=float32)
"""
)

_bayesian_optimize_doc = (
    # % TODO FC: Add advanced user guide
    """
Model bayesian assimilation using numerical optimization algorithms.

Parameters
----------
%(model_parameter)s

"""
    + _gen_docstring_from_base_doc(MAPPING_OPTIMIZER_BASE_DOC, ["mapping", "optimizer"], nindent=0)
    + """

optimize_options : `dict[str, Any]` or None, default None
    Dictionary containing optimization options for fine-tuning the optimization process.
    See `%(default_optimize_options_func)s` to retrieve the default optimize options based on the **mapping**
    and **optimizer**.

"""
    + _gen_docstring_from_base_doc(
        OPTIMIZE_OPTIONS_BASE_DOC,
        BAYESIAN_OPTIMIZE_OPTIONS_KEYS_DOC,
        nindent=1,
    )
    + """
cost_options : `dict[str, Any]` or None, default None
    Dictionary containing computation cost options for simulated and observed responses. The elements are:

"""
    + _gen_docstring_from_base_doc(
        COST_OPTIONS_BASE_DOC,
        DEFAULT_SIMULATION_COST_OPTIONS["bayesian_optimize"].keys(),
        nindent=1,
    )
    + """
common_options : `dict[str, Any]` or None, default None
    Dictionary containing common options with two elements:

"""
    + _gen_docstring_from_base_doc(
        COMMON_OPTIONS_BASE_DOC, DEFAULT_SIMULATION_COMMON_OPTIONS.keys(), nindent=1
    )
    + """
return_options : `dict[str, Any]` or None, default None
    Dictionary containing return options to save additional simulation results. The elements are:

"""
    + _gen_docstring_from_base_doc(
        RETURN_OPTIONS_BASE_DOC,
        DEFAULT_SIMULATION_RETURN_OPTIONS["bayesian_optimize"].keys(),
        nindent=1,
    )
    + """
callback : callable or None, default None
    A callable called after each iteration with the signature ``callback(iopt: BayesianOptimize)``, where
    ``iopt`` is a keyword argument representing an instance of the `BayesianOptimize` class that contains
    intermediate optimization results with attributes:

    - ``'control_vector'``: The current control vector.
    - ``'cost'``: The current cost value.
    - ``'n_iter'``: The current number of iterations performed by the optimizer.
    - ``'projg'``: The current projected gradient, available if using gradient-based optimizers.

    >>> import numpy as np
    >>> iter_cost = []  # to get the cost values through iterations
    >>> def callback_func(iopt, icost=iter_cost):
    ...     icost.append(iopt.cost)
    ...     # save the current control vector value to a text file
    ...     np.savetxt(f"control_iter_{len(icost)}.txt", iopt.control_vector)
    >>> callback = callback_func

    .. note::
        The name of the argument must be ``iopt`` for the callback to be passed as a `BayesianOptimize`
        object.

"""
    + """
Returns
-------
%(model_return)s
bayesian_optimize : `BayesianOptimize` or None, default None
    It returns an object containing additional simulation results with the keys defined in
    **return_options**. If no keys are defined, it returns None.

See Also
--------
BayesianOptimize : Represents bayesian optimize optional results.

Examples
--------
>>> from smash.factory import load_dataset
>>> setup, mesh = load_dataset("cance")
>>> model = smash.Model(setup, mesh)

Optimize the Model

>>> %(model_example_func)s
</> Bayesian Optimize
    At iterate     0    nfg =     1    J = 7.70491e+01    ddx = 0.64
    At iterate     1    nfg =    68    J = 2.58460e+00    ddx = 0.64
    At iterate     2    nfg =   135    J = 2.32432e+00    ddx = 0.32
    At iterate     3    nfg =   202    J = 2.30413e+00    ddx = 0.08
    At iterate     4    nfg =   269    J = 2.26219e+00    ddx = 0.08
    At iterate     5    nfg =   343    J = 2.26025e+00    ddx = 0.01
    At iterate     6    nfg =   416    J = 2.25822e+00    ddx = 0.01
    CONVERGENCE: DDX < 0.01

Get the simulated discharges:

>>> %(model_example_response)s.response.q
array([[3.8725790e-04, 3.5435968e-04, 3.0995542e-04, ..., 1.9623449e+01,
        1.9391096e+01, 1.9163761e+01],
       [9.0669666e-05, 6.3609048e-05, 3.9684954e-05, ..., 4.7896299e+00,
        4.7395458e+00, 4.6904192e+00],
       [1.6136990e-05, 7.8192916e-06, 3.4578943e-06, ..., 1.2418084e+00,
        1.2288600e+00, 1.2161493e+00]], dtype=float32)
"""
)

_multiple_forward_run_doc = (
    # % TODO FC: Add advanced user guide
    """
Run the forward Model with multiple sets of parameters.

Parameters
----------
model : `Model`
    Primary data structure of the hydrological model `smash`.

samples : `Samples` or `dict[str, Any]`
    Represents the rainfall-runoff parameters and/or initial states sample.
    This can be either a `Samples` object or a dictionary, where the keys are parameter/state names
    and the corresponding value is a sequence of specified values, representing multiple samples.

cost_options : `dict[str, Any]` or None, default None
    Dictionary containing computation cost options for simulated and observed responses. The elements are:

"""
    + _gen_docstring_from_base_doc(
        COST_OPTIONS_BASE_DOC,
        DEFAULT_SIMULATION_COST_OPTIONS["forward_run"].keys(),
        nindent=1,
    )
    + """
common_options : `dict[str, Any]` or None, default None
    Dictionary containing common options with two elements:

"""
    + _gen_docstring_from_base_doc(
        COMMON_OPTIONS_BASE_DOC, DEFAULT_SIMULATION_COMMON_OPTIONS.keys(), nindent=1
    )
    + """

Returns
-------
multiple_forward_run : `MultipleForwardRun`
    It returns an object containing the results of the multiple forward run.

See Also
--------
Samples : Represents the generated samples result.
MultipleForwardRun : Represents the multiple forward run result.

Examples
--------
>>> from smash.factory import load_dataset
>>> from smash.factory import generate_samples
>>> setup, mesh = load_dataset("cance")
>>> model = smash.Model(setup, mesh)

Define sampling problem and generate samples

>>> problem = {
...            'num_vars': 4,
...            'names': ['cp', 'ct', 'kexc', 'llr'],
...            'bounds': [[1, 2000], [1, 1000], [-20, 5], [1, 1000]]
... }
>>> sr = generate_samples(problem, n=5, random_state=11)

Run Model with multiple sets of parameters

>>> mfr = smash.multiple_forward_run(model, samples=sr)
</> Multiple Forward Run
    Forward Run 5/5 (100%)

Get the cost values through multiple forward runs

>>> mfr.cost
array([1.2170078, 1.0733036, 1.2239422, 1.2506678, 1.2261102], dtype=float32)
    """
)

_optimize_control_info_doc = (
    """
Information on the optimization control vector of Model.

Parameters
----------
model : `Model`
    Primary data structure of the hydrological model `smash`.

"""
    + _gen_docstring_from_base_doc(MAPPING_OPTIMIZER_BASE_DOC, ["mapping", "optimizer"], nindent=0)
    + """

optimize_options : `dict[str, Any]` or None, default None
    Dictionary containing optimization options for fine-tuning the optimization process.
    See `%(default_optimize_options_func)s` to retrieve the default optimize options based on the **mapping**
    and **optimizer**.

"""
    + _gen_docstring_from_base_doc(
        OPTIMIZE_OPTIONS_BASE_DOC,
        OPTIMIZE_OPTIONS_KEYS_DOC,
        nindent=1,
    )
    + RETURN_CONTROL_INFO_BASE_DOC
    + """

    - name : `numpy.ndarray`
        An array of shape *(n,)* containing the names of the control vector. The naming convention is:

        - ``<key>-0``: Spatially uniform parameter or multi-linear/power intercept where ``<key>`` is the
          name of any rainfall-runoff parameters or initial_states (``'cp-0'``, ``'llr-0'``, ``'ht-0'``, etc).
        - ``<key>-<row>-<col>``: Spatially distributed parameter where ``<key>`` is the name of any
          rainfall-runoff parameters or initial_states and ``<row>``, ``<col>``, the corresponding position in
          the spatial domain (``'cp-1-1'``, ``'llr-20-2'``, ``'ht-3-12'``, etc). It's one based indexing.
        - ``<key>-<desc>-<kind>``: Multi-linear/power descriptor linked parameter where ``<key>`` is the
          name of any rainfall-runoff parameters or initial_states, ``<desc>`` the corresponding descriptor
          and ``<kind>``, the kind of parameter (coefficient or exposant) (``'cp-slope-a'``,
          ``'llr-slope-b'``, ``'ht-dd-a'``).
        - ``<key>-<row>-<col>``: Weights and biases of the parameterization neural network where ``<key>``
          indicates the layer and type of parameter (e.g., ``'weight_1'`` for the first layer weights,
          ``'bias_2'`` for the second layer biases), and ``<row>``, ``<col>`` represent the corresponding
          position in the matrix or vector (``'weight_2-23-21'``, ``'bias_1-16'``, etc).
        - ``reg_<key>-<row>-<col>``: Weights and biases of the regionalization neural network where ``<key>``
          indicates the layer and type of parameter (e.g., ``'reg_weight_1'`` for the first layer weights,
          ``'reg_bias_3'`` for the second layer biases), and ``<row>``, ``<col>`` represent the corresponding
          position in the matrix or vector (``'reg_weight_3-32-28'``, ``'reg_bias_1-4'``, etc).

Examples
--------
>>> from smash.factory import load_dataset
>>> setup, mesh = load_dataset("cance")
>>> model = smash.Model(setup, mesh)

Default optimize control vector information

>>> control_info = smash.optimize_control_info(model)
>>> control_info
{
    'l': array([-13.815511 , -13.815511 ,  -4.6052704, -13.815511 ], dtype=float32),
    'l_raw': array([ 1.e-06,  1.e-06, -5.e+01,  1.e-06], dtype=float32),
    'n': 4,
    'name': array(['cp-0', 'ct-0', 'kexc-0', 'llr-0'], dtype='<U128'),
    'nbd': array([2, 2, 2, 2], dtype=int32),
    'nbk': array([4, 0, 0, 0, 0, 0]),
    'u': array([6.9077554, 6.9077554, 4.6052704, 6.9077554], dtype=float32),
    'u_raw': array([1000., 1000.,   50., 1000.], dtype=float32),
    'x': array([5.2983174, 6.214608 , 0.       , 1.609438 ], dtype=float32),
    'x_raw': array([200., 500.,   0.,   5.], dtype=float32),
}

This gives a direct indication of what the optimizer takes as input, depending on the optimization
configuration set up. 4 rainfall-runoff parameters are uniformly optimized (``'cp-0'``, ``'ct-0'``,
``'kexc-0'`` and ``'llr-0'``). Each parameter has a lower and upper bound (``2`` in ``nbd``) and a
transformation was applied to the control (``x`` relative to ``x_raw``).

With a customize optimize configuration. Here, choosing a ``multi-linear`` mapping and optimizing only ``cp``
and ``kexc`` with different descriptors

>>> control_info = smash.optimize_control_info(
        model,
        mapping="multi-linear",
        optimize_options={
            "parameters": ["cp", "kexc"],
            "descriptor": {"kexc": ["dd"]},
        },
    )
>>> control_info
{
    'l': array([-inf, -inf, -inf, -inf, -inf], dtype=float32),
    'l_raw': array([-inf, -inf, -inf, -inf, -inf], dtype=float32),
    'n': 5,
    'name': array(['cp-0', 'cp-slope-a', 'cp-dd-a', 'kexc-0', 'kexc-dd-a'], dtype='<U128'),
    'nbd': array([0, 0, 0, 0, 0], dtype=int32),
    'nbk': array([5, 0, 0, 0, 0, 0]),
    'u': array([inf, inf, inf, inf, inf], dtype=float32),
    'u_raw': array([inf, inf, inf, inf, inf], dtype=float32),
    'x': array([-1.3862944,  0.       ,  0.       ,  0.       ,  0.       ], dtype=float32),
    'x_raw': array([-1.3862944,  0.       ,  0.       ,  0.       ,  0.       ], dtype=float32),
}

5 parameters are optimized which are the intercepts (``'cp-0'`` and  ``'kexc-0'``) and the coefficients
(``'cp-slope-a'``, ``'cp-dd-a'`` and ``'kexc-dd-a'``) of the regression between the descriptors
(``slope`` and ``dd``) and the rainfall-runoff parameters (``cp`` and ``kexc``).
"""
)

_bayesian_optimize_control_info_doc = (
    """
Information on the bayesian optimization control vector of Model.

Parameters
----------
model : `Model`
    Primary data structure of the hydrological model `smash`.

"""
    + _gen_docstring_from_base_doc(MAPPING_OPTIMIZER_BASE_DOC, ["mapping", "optimizer"], nindent=0)
    + """

optimize_options : `dict[str, Any]` or None, default None
    Dictionary containing optimization options for fine-tuning the optimization process.
    See `%(default_optimize_options_func)s` to retrieve the default optimize options based on the **mapping**
    and **optimizer**.

"""
    + _gen_docstring_from_base_doc(
        OPTIMIZE_OPTIONS_BASE_DOC,
        BAYESIAN_OPTIMIZE_OPTIONS_KEYS_DOC,
        nindent=1,
    )
    + """
cost_options : `dict[str, Any]` or None, default None
    Dictionary containing computation cost options for simulated and observed responses. The elements are:

"""
    + _gen_docstring_from_base_doc(
        COST_OPTIONS_BASE_DOC,
        DEFAULT_SIMULATION_COST_OPTIONS["bayesian_optimize"].keys(),
        nindent=1,
    )
    + RETURN_CONTROL_INFO_BASE_DOC
    + """

    - name : `numpy.ndarray`
        An array of shape *(n,)* containing the names of the control vector. The naming convention is:

        - ``<key>-0``: Spatially uniform parameter or multi-linear/power intercept where ``<key>`` is the
          name of any rainfall-runoff parameters or initial_states (``'cp-0'``, ``'llr-0'``, ``'ht-0'``, etc).
        - ``<key>-<row>-<col>``: Spatially distributed parameter where ``<key>`` is the name of any
          rainfall-runoff parameters or initial_states and ``<row>``, ``<col>``, the corresponding position in
          the spatial domain (``'cp-1-1'``, ``'llr-20-2'``, ``'ht-3-12'``, etc). It's one based indexing.
        - ``<key>-<desc>-<kind>``: Multi-linear/power descriptor linked parameter where ``<key>`` is the
          name of any rainfall-runoff parameters or initial_states, ``<desc>`` the corresponding descriptor
          and ``<kind>``, the kind of parameter (coefficient or exposant) (``'cp-slope-a'``,
          ``'llr-slope-b'``, ``'ht-dd-a'``).
        - ``<key>-<code>``: Structural error parameter where ``<key>`` is the name of any structural error mu
          or sigma parameters and ``<code>``, the corresponding gauge (``'sg0-V3524010'``, ``'sg1-V3524010'``,
          etc).
        - ``<key>-<row>-<col>``: Weights and biases of the parameterization neural network where ``<key>``
          indicates the layer and type of parameter (e.g., ``'weight_1'`` for the first layer weights,
          ``'bias_2'`` for the second layer biases), and ``<row>``, ``<col>`` represent the corresponding
          position in the matrix or vector (``'weight_2-23-21'``, ``'bias_1-16'``, etc).

Examples
--------
>>> from smash.factory import load_dataset
>>> setup, mesh = load_dataset("cance")
>>> model = smash.Model(setup, mesh)

Default optimize control vector information

>>> control_info = smash.bayesian_optimize_control_info(model)
>>> control_info
{
    'l': array([-1.3815511e+01, -1.3815511e+01, -4.6052704e+00, -1.3815511e+01, 1.0000000e-06, 1.0000000e-06],
         dtype=float32),
    'l_raw': array([ 1.e-06,  1.e-06, -5.e+01,  1.e-06,  1.e-06,  1.e-06], dtype=float32),
    'n': 6,
    'name': array(['cp-0', 'ct-0', 'kexc-0', 'llr-0', 'sg0-V3524010', 'sg1-V3524010'], dtype='<U128'),
    'nbd': array([2, 2, 2, 2, 2, 2], dtype=int32),
    'nbk': array([4, 0, 0, 2]),
    'u': array([   6.9077554,    6.9077554,    4.6052704,    6.9077554, 1000.       ,   10.       ],
         dtype=float32),
    'u_raw': array([1000., 1000.,   50., 1000., 1000.,   10.], dtype=float32),
    'x': array([5.2983174, 6.214608 , 0.       , 1.609438 , 1.       , 0.2      ], dtype=float32),
    'x_raw': array([2.e+02, 5.e+02, 0.e+00, 5.e+00, 1.e+00, 2.e-01], dtype=float32),
}

This gives a direct indication of what the optimizer takes as input, depending on the optimization
configuration set up. 4 rainfall-runoff parameters are uniformly optimized (``'cp-0'``, ``'ct-0'``,
``'kexc-0'`` and ``'llr-0'``) and 2 structural error sigma parameters at gauge ``'V3524010'``
(``'sg0-V3524010'``, ``'sg1-V3524010'``). Each parameter has a lower and upper bound (``2`` in ``nbd``)
and a transformation was applied to the control (``x`` relative to ``x_raw``).

With a customize optimize configuration. Here, choosing a ``multi-linear`` mapping and
optimizing only 2 rainfall-runoff parameters ``cp``, ``kexc`` with different descriptors and 2 structural
error sigma parameters ``sg0`` and ``sg1``.

>>> control_info = smash.bayesian_optimize_control_info(
        model,
        mapping="multi-linear",
        optimize_options={
            "parameters": ["cp", "kexc", "sg0", "sg1"],
            "descriptor": {"kexc": ["dd"]},
        },
    )
>>> control_info
{
    'l': array([-inf, -inf, -inf, -inf, -inf,   0.,   0.], dtype=float32),
    'l_raw': array([  -inf,   -inf,   -inf,   -inf,   -inf, 1.e-06, 1.e-06], dtype=float32),
    'n': 7,
    'name': array(['cp-0', 'cp-slope-a', 'cp-dd-a', 'kexc-0', 'kexc-dd-a', 'sg0-V3524010', 'sg1-V3524010'],
            dtype='<U128'),
    'nbd': array([0, 0, 0, 0, 0, 2, 2], dtype=int32),
    'nbk': array([5, 0, 0, 2]),
    'u': array([inf, inf, inf, inf, inf,  1.,  1.], dtype=float32),
    'u_raw': array([  inf,   inf,   inf,   inf,   inf, 1000.,   10.], dtype=float32),
    'x': array([-1.3862944e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00, 0.0000000e+00,  9.9999900e-04,
         1.9999903e-02], dtype=float32),
    'x_raw': array([-1.3862944,  0.       ,  0.       ,  0.       ,  0.       , 1.       ,  0.2      ],
             dtype=float32),
}

7 parameters are optimized which are the intercepts (``'cp-0'`` and  ``'kexc-0'``)
and the coefficients (``'cp-slope-a'``, ``'cp-dd-a'`` and ``'kexc-dd-a'``) of the regression between the
descriptors  (``slope`` and ``dd``) and the rainfall-runoff parameters (``cp`` and ``kexc``) and the 2
structural error sigma parameters (``'sg0'`` and ``'sg1'``) associated to the gauge ``'V3524010'``.

Retrieving information from the control vector is particularly useful for defining priors on the parameters.
During a bayesian optimization, it is possible to define these priors in the **cost_options** argument within
the ``'control_prior'`` key. The problem is that we don't know the control vector in advance until we've
filled in all the optimization options. This is why we can define all the optimization options in the
`bayesian_optimize_control_info` method, retrieve the names of the
parameters that make up the control vector and then call the optimization function, assigning the priors we
want to.

Assign Gaussian priors to the two rainfall-runoff parameters ``'cp-0'`` and ``'kexc-0'`` and perform
a spatially uniform optimization

>>> model.bayesian_optimize(
        cost_options={
            "control_prior": {"cp-0": ["Gaussian", [200, 100]], "kexc-0": ["Gaussian", [0, 5]]}
        },
    )
</> Bayesian Optimize
    At iterate     0    nfg =     1    J = 8.05269e+01    ddx = 0.64
    At iterate     1    nfg =    68    J = 3.02925e+00    ddx = 0.64
    At iterate     2    nfg =   135    J = 2.76492e+00    ddx = 0.32
    At iterate     3    nfg =   203    J = 2.76056e+00    ddx = 0.04
    At iterate     4    nfg =   271    J = 2.75504e+00    ddx = 0.02
    At iterate     5    nfg =   344    J = 2.75420e+00    ddx = 0.01
    At iterate     6    nfg =   392    J = 2.75403e+00    ddx = 0.01
    CONVERGENCE: DDX < 0.01
"""
)

_default_optimize_options_doc = (
    """
Default optimization options of Model.

Parameters
----------
model : `Model`
    Primary data structure of the hydrological model `smash`.

"""
    + _gen_docstring_from_base_doc(MAPPING_OPTIMIZER_BASE_DOC, ["mapping", "optimizer"], nindent=0)
    + """

Returns
-------
optimize_options : `dict[str, Any]`
    Dictionary containing optimization options for fine-tuning the optimization process. The specific keys
    returned depend on the chosen **mapping** and **optimizer**. This dictionary can be directly passed to
    the **optimize_options** argument of the `optimize` (or `Model.optimize`) method.

Examples
--------
>>> from smash.factory import load_dataset
>>> setup, mesh = load_dataset("cance")
>>> model = smash.Model(setup, mesh)

Get the default optimization options for ``'uniform'`` mapping

>>> opt_u = smash.default_optimize_options(model, mapping="uniform")
>>> opt_u
{
    'parameters': ['cp', 'ct', 'kexc', 'llr'],
    'bounds': {
        'cp': (1e-06, 1000.0),
        'ct': (1e-06, 1000.0),
        'kexc': (-50, 50),
        'llr': (1e-06, 1000.0)
    },
    'control_tfm': 'sbs',
    'termination_crit': {'maxiter': 50},
}

Directly pass this dictionary to the **optimize_options** argument of the `optimize`
(or `Model.optimize`) method.
It's equivalent to set **optimize_options** to None (which is the default value)

>>> model_u = smash.optimize(model, mapping="uniform", optimize_options=opt_u)
</> Optimize
    At iterate     0    nfg =     1    J = 6.95010e-01    ddx = 0.64
    At iterate     1    nfg =    30    J = 9.84107e-02    ddx = 0.64
    At iterate     2    nfg =    59    J = 4.54087e-02    ddx = 0.32
    At iterate     3    nfg =    88    J = 3.81818e-02    ddx = 0.16
    At iterate     4    nfg =   117    J = 3.73617e-02    ddx = 0.08
    At iterate     5    nfg =   150    J = 3.70873e-02    ddx = 0.02
    At iterate     6    nfg =   183    J = 3.68004e-02    ddx = 0.02
    At iterate     7    nfg =   216    J = 3.67635e-02    ddx = 0.01
    At iterate     8    nfg =   240    J = 3.67277e-02    ddx = 0.01
    CONVERGENCE: DDX < 0.01

Customize the optimization options by removing ``'kexc'`` from the optimized parameters

>>> opt_u["parameters"].remove("kexc")
>>> opt_u
{
    'parameters': ['cp', 'ct', 'llr'],
    'bounds': {
        'cp': (1e-06, 1000.0),
        'ct': (1e-06, 1000.0),
        'kexc': (-50, 50),
        'llr': (1e-06, 1000.0)
    },
    'control_tfm': 'sbs',
    'termination_crit': {'maxiter': 50},
}

Run the optimization method

>>> model_u = smash.optimize(model, mapping="uniform", optimize_options=opt_u)
ValueError: Unknown, non optimized, or unbounded parameter 'kexc' in bounds optimize_options.
Choices: ['cp', 'ct', 'llr']

An error is raised because we define ``bounds`` to a non optimized parameter ``kexc``. Remove also
``kexc`` from bounds

>>> opt_u["bounds"].pop("kexc")
(-50, 50)

.. note::
    The built-in dictionary method `pop <https://docs.python.org/3/library/stdtypes.html#dict.pop>`__
    returns the value associated to the removed key

>>> opt_u
{
    'parameters': ['cp', 'ct', 'llr'],
    'bounds': {
        'cp': (1e-06, 1000.0),
        'ct': (1e-06, 1000.0),
        'llr': (1e-06, 1000.0)
    },
    'control_tfm': 'sbs',
    'termination_crit': {'maxiter': 50},
}

Run again the optimization to see the differences linked to a change in control vector

>>> model_u = smash.optimize(model, mapping="uniform", optimize_options=opt_u)
</> Optimize
    At iterate     0    nfg =     1    J = 6.95010e-01    ddx = 0.64
    At iterate     1    nfg =    17    J = 1.28863e-01    ddx = 0.64
    At iterate     2    nfg =    32    J = 6.94838e-02    ddx = 0.32
    At iterate     3    nfg =    49    J = 4.50720e-02    ddx = 0.16
    At iterate     4    nfg =    65    J = 4.40468e-02    ddx = 0.08
    At iterate     5    nfg =    84    J = 4.35278e-02    ddx = 0.04
    At iterate     6    nfg =   102    J = 4.26906e-02    ddx = 0.02
    At iterate     7    nfg =   122    J = 4.26645e-02    ddx = 0.01
    At iterate     8    nfg =   140    J = 4.26062e-02    ddx = 0.01
    CONVERGENCE: DDX < 0.01

Get the default optimization options for a different mapping

>>> opt_ann = smash.default_optimize_options(model, mapping="ann")
>>> opt_ann
{
    'parameters': ['cp', 'ct', 'kexc', 'llr'],
    'bounds': {
        'cp': (1e-06, 1000.0),
        'ct': (1e-06, 1000.0),
        'kexc': (-50, 50),
        'llr': (1e-06, 1000.0)
    },
    'net':
        +----------------------------------------------------------+
        | Layer Type            Input/Output Shape  Num Parameters |
        +----------------------------------------------------------+
        | Dense                 (2,)/(18,)          54             |
        | Activation (ReLU)     (18,)/(18,)         0              |
        | Dense                 (18,)/(9,)          171            |
        | Activation (ReLU)     (9,)/(9,)           0              |
        | Dense                 (9,)/(4,)           40             |
        | Activation (Sigmoid)  (4,)/(4,)           0              |
        | Scale (MinMaxScale)   (4,)/(4,)           0              |
        +----------------------------------------------------------+
        Total parameters: 265
        Trainable parameters: 265,
    'learning_rate': 0.003,
    'random_state': None,
    'termination_crit': {'maxiter': 200, 'early_stopping': 0}
}

Again, customize the optimization options and optimize the Model

>>> opt_ann["learning_rate"] = 0.006
>>> opt_ann["termination_crit"]["maxiter"] = 50
>>> opt_ann["termination_crit"]["early_stopping"] = 5
>>> opt_ann["random_state"] = 21
>>> model.optimize(mapping="ann", optimize_options=opt_ann)
</> Optimize
    At iterate     0    nfg =     1    J = 1.22206e+00    |proj g| = 2.09135e-04
    At iterate     1    nfg =     2    J = 1.21931e+00    |proj g| = 2.39937e-04
    ...
    At iterate    40    nfg =    41    J = 5.21514e-02    |proj g| = 1.31863e-02
    At iterate    41    nfg =    42    J = 5.12064e-02    |proj g| = 3.74748e-03
    At iterate    42    nfg =    43    J = 5.79208e-02    |proj g| = 5.08674e-03
    At iterate    43    nfg =    44    J = 6.38050e-02    |proj g| = 1.01001e-02
    At iterate    44    nfg =    45    J = 6.57343e-02    |proj g| = 1.33649e-02
    At iterate    45    nfg =    46    J = 6.45393e-02    |proj g| = 1.56155e-02
    At iterate    46    nfg =    47    J = 6.33092e-02    |proj g| = 1.72698e-02
    EARLY STOPPING: NO IMPROVEMENT for 5 CONSECUTIVE ITERATIONS
    Revert to iteration 41 with J = 5.12064e-02 due to early stopping

The training process was terminated after 46 iterations, where the loss did not decrease below the minimal
value at iteration 41 for 5 consecutive iterations. The optimal parameters are thus recorded at iteration 41.
"""
)

_default_bayesian_optimize_options_doc = (
    """
Default bayesian optimization options of Model.

Parameters
----------
model : `Model`
    Primary data structure of the hydrological model `smash`.

"""
    + _gen_docstring_from_base_doc(MAPPING_OPTIMIZER_BASE_DOC, ["mapping", "optimizer"], nindent=0)
    + """

Returns
-------
optimize_options : `dict[str, Any]`
    Dictionary containing optimization options for fine-tuning the optimization process. The specific keys
    returned depend on the chosen **mapping** and **optimizer**. This dictionary can be directly passed to
    the **optimize_options** argument of the `bayesian_optimize` (or `Model.bayesian_optimize`) method.

Examples
--------
>>> from smash.factory import load_dataset
>>> setup, mesh = load_dataset("cance")
>>> model = smash.Model(setup, mesh)

Get the default bayesian optimization options for ``'uniform'`` mapping

>>> opt_u = smash.default_bayesian_optimize_options(model, mapping="uniform")
>>> opt_u
{
    'parameters': ['cp', 'ct', 'kexc', 'llr', 'sg0', 'sg1'],
    'bounds': {
        'cp': (1e-06, 1000.0),
        'ct': (1e-06, 1000.0),
        'kexc': (-50, 50),
        'llr': (1e-06, 1000.0),
        'sg0': (1e-06, 1000.0),
        'sg1': (1e-06, 10.0)
    },
    'control_tfm': 'sbs',
    'termination_crit': {'maxiter': 50},
}

Directly pass this dictionary to the **optimize_options** argument of the `bayesian_optimize`
(or `Model.bayesian_optimize`) method.
It's equivalent to set **optimize_options** to None (which is the default value)

>>> model_u = smash.bayesian_optimize(model, mapping="uniform", optimize_options=opt_u)
</> Bayesian Optimize
    At iterate     0    nfg =     1    J = 7.70491e+01    ddx = 0.64
    At iterate     1    nfg =    68    J = 2.58460e+00    ddx = 0.64
    At iterate     2    nfg =   135    J = 2.32432e+00    ddx = 0.32
    At iterate     3    nfg =   202    J = 2.30413e+00    ddx = 0.08
    At iterate     4    nfg =   269    J = 2.26219e+00    ddx = 0.08
    At iterate     5    nfg =   343    J = 2.26025e+00    ddx = 0.01
    At iterate     6    nfg =   416    J = 2.25822e+00    ddx = 0.01
    CONVERGENCE: DDX < 0.01

Get the default bayesian optimization options for a different mapping

>>> opt_ml = smash.default_bayesian_optimize_options(model, mapping="multi-linear")
>>> opt_ml
{
    'parameters': ['cp', 'ct', 'kexc', 'llr', 'sg0', 'sg1'],
    'bounds': {
        'cp': (1e-06, 1000.0),
        'ct': (1e-06, 1000.0),
        'kexc': (-50, 50),
        'llr': (1e-06, 1000.0),
        'sg0': (1e-06, 1000.0),
        'sg1': (1e-06, 10.0)
    },
    'control_tfm': 'normalize',
    'descriptor': {
        'cp': array(['slope', 'dd'], dtype='<U5'),
        'ct': array(['slope', 'dd'], dtype='<U5'),
        'kexc': array(['slope', 'dd'], dtype='<U5'),
        'llr': array(['slope', 'dd'], dtype='<U5')
    },
    'termination_crit': {'maxiter': 100, 'factr': 1000000.0, 'pgtol': 1e-12},
}

Customize the bayesian optimization options and optimize the Model

>>> opt_ml["bounds"]["cp"] = (1, 2000)
>>> opt_ml["bounds"]["sg0"] = (1e-3, 100)
>>> opt_ml["descriptor"]["cp"] = "slope"
>>> opt_ml["termination_crit"]["maxiter"] = 10
>>> model.bayesian_optimize(mapping="multi-linear", optimize_options=opt_ml)
</> Bayesian Optimize
    At iterate     0    nfg =     1    J = 7.70491e+01    |proj g| = 1.05147e+04
    At iterate     1    nfg =     2    J = 6.69437e+00    |proj g| = 2.15263e+02
    At iterate     2    nfg =     3    J = 6.52716e+00    |proj g| = 2.03207e+02
    At iterate     3    nfg =     4    J = 5.08876e+00    |proj g| = 6.83760e+01
    At iterate     4    nfg =     5    J = 4.73664e+00    |proj g| = 4.19148e+01
    At iterate     5    nfg =     6    J = 4.42125e+00    |proj g| = 1.94103e+01
    At iterate     6    nfg =     7    J = 4.28494e+00    |proj g| = 9.39774e+00
    At iterate     7    nfg =     8    J = 4.19646e+00    |proj g| = 4.74194e+00
    At iterate     8    nfg =     9    J = 4.13953e+00    |proj g| = 1.74698e+00
    At iterate     9    nfg =    10    J = 4.09997e+00    |proj g| = 1.04288e+00
    At iterate    10    nfg =    11    J = 4.02741e+00    |proj g| = 4.41394e+00
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT

The optimization process was terminated after 10 iterations, the maximal value we defined.
"""
)

_set_control_optimize_doc = (
    """
Retrieve the Model parameters/states from the optimization control vector.

Parameters
----------
control_vector : `numpy.ndarray`
    A 1D array representing the control values, ideally obtained from the `optimize` (or `Model.optimize`)
    method.

"""
    + _gen_docstring_from_base_doc(MAPPING_OPTIMIZER_BASE_DOC, ["mapping", "optimizer"], nindent=0)
    + """

optimize_options : `dict[str, Any]` or None, default None
    Dictionary containing optimization options for fine-tuning the optimization process.
    See `%(default_optimize_options_func)s` to retrieve the default optimize options based on the **mapping**
    and **optimizer**.

"""
    + _gen_docstring_from_base_doc(
        OPTIMIZE_OPTIONS_BASE_DOC,
        OPTIMIZE_OPTIONS_KEYS_DOC,
        nindent=1,
    )
    + """

Examples
--------
>>> from smash.factory import load_dataset
>>> setup, mesh = load_dataset("cance")
>>> model = smash.Model(setup, mesh)

Define a callback function to store the control vector solutions during the optimization process

>>> iter_control = []
>>> def callback(iopt, icontrol=iter_control):
...     icontrol.append(iopt.control_vector)

Optimize the Model

>>> model.optimize(callback=callback)

Retrieve the model parameters from the control vector solution at the first iteration

>>> model.set_control_optimize(iter_control[0])

Perform a forward run to update the hydrological responses and final states

>>> model.forward_run()
"""
)

_set_control_bayesian_optimize_doc = (
    """
Retrieve the Model parameters/states from the bayesian optimization control vector.

Parameters
----------
control_vector : `numpy.ndarray`
    A 1D array representing the control values, ideally obtained from the `bayesian_optimize`
    (or `Model.bayesian_optimize`) method.

"""
    + _gen_docstring_from_base_doc(MAPPING_OPTIMIZER_BASE_DOC, ["mapping", "optimizer"], nindent=0)
    + """

optimize_options : `dict[str, Any]` or None, default None
    Dictionary containing optimization options for fine-tuning the optimization process.
    See `%(default_optimize_options_func)s` to retrieve the default optimize options based on the **mapping**
    and **optimizer**.

"""
    + _gen_docstring_from_base_doc(
        OPTIMIZE_OPTIONS_BASE_DOC,
        BAYESIAN_OPTIMIZE_OPTIONS_KEYS_DOC,
        nindent=1,
    )
    + """
cost_options : `dict[str, Any]` or None, default None
    Dictionary containing computation cost options for simulated and observed responses. The elements are:

"""
    + _gen_docstring_from_base_doc(
        COST_OPTIONS_BASE_DOC,
        DEFAULT_SIMULATION_COST_OPTIONS["bayesian_optimize"].keys(),
        nindent=1,
    )
    + """

Examples
--------
>>> from smash.factory import load_dataset
>>> setup, mesh = load_dataset("cance")
>>> model = smash.Model(setup, mesh)

Define a callback function to store the control vector solutions during the optimization process

>>> iter_control = []
>>> def callback(iopt, icontrol=iter_control):
...     icontrol.append(iopt.control_vector)

Optimize the Model

>>> model.bayesian_optimize(callback=callback)

Retrieve the model parameters from the control vector solution at the first iteration

>>> model.set_control_bayesian_optimize(iter_control[0])

Perform a forward run to update the hydrological responses and final states

>>> model.forward_run()
"""
)

_forward_run_doc_appender = DocAppender(_forward_run_doc, indents=0)
_smash_forward_run_doc_substitution = DocSubstitution(
    model_parameter="model : `Model`\n\tPrimary data structure of the hydrological model `smash`.",
    model_return="model : `Model`\n\t It returns an updated copy of the initial Model object.",
    model_example_func="model_fwd = smash.forward_run()",
    model_example_response="model_fwd",
    percent="%",
)
_model_forward_run_doc_substitution = DocSubstitution(
    model_parameter="",
    model_return="",
    model_example_func="model.forward_run()",
    model_example_response="model",
    percent="%",
)

_optimize_doc_appender = DocAppender(_optimize_doc, indents=0)
_smash_optimize_doc_substitution = DocSubstitution(
    model_parameter="model : `Model`\n\tPrimary data structure of the hydrological model `smash`.",
    mapping_ann="- ``'ann'``",
    optimizer_lbfgsb="- ``'lbfgsb'`` (for all mappings except ``'ann'``)",
    default_optimizer_for_ann_mapping="- ``'adam'`` for **mapping** = ``'ann'``",
    default_optimize_options_func="default_optimize_options",
    parameters_serr_mu_parameters="",
    parameters_serr_sigma_parameters="",
    parameters_note_serr_parameters="",
    bounds_get_serr_parameters_bounds="",
    model_return="model : `Model`\n\t It returns an updated copy of the initial Model object.",
    model_example_func="model_opt = smash.optimize()",
    model_example_response="model_opt",
    percent="%",
)
_model_optimize_doc_substitution = DocSubstitution(
    model_parameter="",
    mapping_ann="- ``'ann'``",
    optimizer_lbfgsb="- ``'lbfgsb'`` (for all mappings except ``'ann'``)",
    default_optimizer_for_ann_mapping="- ``'adam'`` for **mapping** = ``'ann'``",
    default_optimize_options_func="default_optimize_options",
    parameters_serr_mu_parameters="",
    parameters_serr_sigma_parameters="",
    parameters_note_serr_parameters="",
    bounds_get_serr_parameters_bounds="",
    model_return="",
    model_example_func="model.optimize()",
    model_example_response="model",
    percent="%",
)

_multiset_estimate_doc_appender = DocAppender(_multiset_estimate_doc, indents=0)
_smash_multiset_estimate_doc_substitution = DocSubstitution(
    model_parameter="model : `Model`\n\tPrimary data structure of the hydrological model `smash`.",
    model_return="model : `Model`\n\t It returns an updated copy of the initial Model object.",
    model_example_func="model_estim = smash.multiset_estimate(model, multiset=mfr)",
    model_example_response="model_estim",
    percent="%",
)
_model_multiset_estimate_doc_substitution = DocSubstitution(
    model_parameter="",
    model_return="",
    model_example_func="model.multiset_estimate(multiset=mfr)",
    model_example_response="model",
    percent="%",
)

_bayesian_optimize_doc_appender = DocAppender(_bayesian_optimize_doc, indents=0)
_smash_bayesian_optimize_doc_substitution = DocSubstitution(
    model_parameter="model : `Model`\n\tPrimary data structure of the hydrological model `smash`.",
    mapping_ann="",
    optimizer_lbfgsb="- ``'lbfgsb'`` (for all mappings)",
    default_optimizer_for_ann_mapping="",
    default_optimize_options_func="default_bayesian_optimize_options",
    parameters_serr_mu_parameters="- `Model.serr_mu_parameters`",
    parameters_serr_sigma_parameters="- `Model.serr_sigma_parameters`",
    parameters_note_serr_parameters=", `Model.serr_mu_parameters`, `Model.serr_sigma_parameters`",
    bounds_get_serr_parameters_bounds=", `Model.get_serr_mu_parameters_bounds` and "
    "`Model.get_serr_sigma_parameters_bounds`",
    model_return="model : `Model`\n\t It returns an updated copy of the initial Model object.",
    model_example_func="model_bayes_opt = smash.bayesian_optimize()",
    model_example_response="model_bayes_opt",
    percent="%",
)
_model_bayesian_optimize_doc_substitution = DocSubstitution(
    model_parameter="",
    mapping_ann="",
    optimizer_lbfgsb="- ``'lbfgsb'`` (for all mappings)",
    default_optimizer_for_ann_mapping="",
    default_optimize_options_func="default_bayesian_optimize_options",
    parameters_serr_mu_parameters="- `Model.serr_mu_parameters`",
    parameters_serr_sigma_parameters="- `Model.serr_sigma_parameters`",
    parameters_note_serr_parameters=", `Model.serr_mu_parameters`, `Model.serr_sigma_parameters`",
    bounds_get_serr_parameters_bounds=", `Model.get_serr_mu_parameters_bounds` and "
    "`Model.get_serr_sigma_parameters_bounds`",
    model_return="",
    model_example_func="model.bayesian_optimize()",
    model_example_response="model",
    percent="%",
)

_multiple_forward_run_doc_appender = DocAppender(_multiple_forward_run_doc, indents=0)

_optimize_control_info_doc_appender = DocAppender(_optimize_control_info_doc, indents=0)
_smash_optimize_control_info_doc_substitution = DocSubstitution(
    mapping_ann="- ``'ann'``",
    optimizer_lbfgsb="- ``'lbfgsb'`` (for all mappings except ``'ann'``)",
    default_optimizer_for_ann_mapping="- ``'adam'`` for **mapping** = ``'ann'``",
    default_optimize_options_func="default_optimize_options",
    parameters_serr_mu_parameters="",
    parameters_serr_sigma_parameters="",
    parameters_note_serr_parameters="",
    bounds_get_serr_parameters_bounds="",
)

_bayesian_optimize_control_info_doc_appender = DocAppender(_bayesian_optimize_control_info_doc, indents=0)
_smash_bayesian_optimize_control_info_doc_substitution = DocSubstitution(
    mapping_ann="",
    optimizer_lbfgsb="- ``'lbfgsb'`` (for all mappings)",
    default_optimizer_for_ann_mapping="",
    default_optimize_options_func="default_bayesian_optimize_options",
    parameters_serr_mu_parameters="- `Model.serr_mu_parameters`",
    parameters_serr_sigma_parameters="- `Model.serr_sigma_parameters`",
    parameters_note_serr_parameters=", `Model.serr_mu_parameters`, `Model.serr_sigma_parameters`",
    bounds_get_serr_parameters_bounds=", `Model.get_serr_mu_parameters_bounds` and "
    "`Model.get_serr_sigma_parameters_bounds`",
)

_set_control_optimize_doc_appender = DocAppender(_set_control_optimize_doc, indents=0)
_set_control_optimize_doc_substitution = DocSubstitution(
    mapping_ann="- ``'ann'``",
    optimizer_lbfgsb="- ``'lbfgsb'`` (for all mappings except ``'ann'``)",
    default_optimizer_for_ann_mapping="- ``'adam'`` for **mapping** = ``'ann'``",
    default_optimize_options_func="default_optimize_options",
    parameters_serr_mu_parameters="",
    parameters_serr_sigma_parameters="",
    parameters_note_serr_parameters="",
    bounds_get_serr_parameters_bounds="",
)

_set_control_bayesian_optimize_doc_appender = DocAppender(_set_control_bayesian_optimize_doc, indents=0)
_set_control_bayesian_optimize_doc_substitution = DocSubstitution(
    mapping_ann="",
    optimizer_lbfgsb="- ``'lbfgsb'`` (for all mappings)",
    default_optimizer_for_ann_mapping="",
    default_optimize_options_func="default_bayesian_optimize_options",
    parameters_serr_mu_parameters="- `Model.serr_mu_parameters`",
    parameters_serr_sigma_parameters="- `Model.serr_sigma_parameters`",
    parameters_note_serr_parameters=", `Model.serr_mu_parameters`, `Model.serr_sigma_parameters`",
    bounds_get_serr_parameters_bounds=", `Model.get_serr_mu_parameters_bounds` and "
    "`Model.get_serr_sigma_parameters_bounds`",
)

_default_optimize_options_doc_appender = DocAppender(_default_optimize_options_doc, indents=0)
_smash_default_optimize_options_doc_substitution = DocSubstitution(
    mapping_ann="- ``'ann'``",
    optimizer_lbfgsb="- ``'lbfgsb'`` (for all mappings except ``'ann'``)",
    default_optimizer_for_ann_mapping="- ``'adam'`` for **mapping** = ``'ann'``",
)

_default_bayesian_optimize_options_doc_appender = DocAppender(
    _default_bayesian_optimize_options_doc, indents=0
)
_smash_default_bayesian_optimize_options_doc_substitution = DocSubstitution(
    mapping_ann="",
    optimizer_lbfgsb="- ``'lbfgsb'`` (for all mappings)",
    default_optimizer_for_ann_mapping="",
)
