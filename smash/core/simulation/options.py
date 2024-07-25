from __future__ import annotations

from typing import TYPE_CHECKING

from smash.core.simulation._standardize import (
    _standardize_default_bayesian_optimize_options_args,
    _standardize_default_optimize_options_args,
    _standardize_simulation_optimize_options,
)

if TYPE_CHECKING:
    from typing import Any

    from smash.core.model.model import Model

__all__ = ["default_optimize_options", "default_bayesian_optimize_options"]


def default_optimize_options(
    model: Model,
    mapping: str = "uniform",
    optimizer: str | None = None,
) -> dict[str, Any]:
    """
    Default optimize options of Model.

    Parameters
    ----------
    model : `Model`
        Primary data structure of the hydrological model `smash`.

    mapping : `str`, default 'uniform'
        Type of mapping. Should be one of

        - ``'uniform'``
        - ``'distributed'``
        - ``'multi-linear'``
        - ``'multi-polynomial'``
        - ``'ann'``

        .. hint::
            See the :ref:`math_num_documentation.mapping` section

    optimizer : `str` or None, default None
        Name of optimizer. Should be one of

        - ``'sbs'`` (``'uniform'`` **mapping** only)
        - ``'lbfgsb'`` (``'uniform'``, ``'distributed'``, ``'multi-linear'`` or ``'multi-polynomial'``
          **mapping** only)
        - ``'sgd'`` (``'ann'`` **mapping** only)
        - ``'adam'`` (``'ann'`` **mapping** only)
        - ``'adagrad'`` (``'ann'`` **mapping** only)
        - ``'rmsprop'`` (``'ann'`` **mapping** only)

        .. note::
            If not given, a default optimizer will be set depending on the optimization mapping:

            - **mapping** = ``'uniform'``; **optimizer** = ``'sbs'``
            - **mapping** = ``'distributed'``, ``'multi-linear'``, or ``'multi-polynomial'``; **optimizer** =
              ``'lbfgsb'``
            - **mapping** = ``'ann'``; **optimizer** = ``'adam'``

        .. hint::
            See the :ref:`math_num_documentation.optimization_algorithm` section

    Returns
    -------
    optimize_options : `dict[str, Any]`
        Dictionary containing optimization options for fine-tuning the optimization process. The specific keys
        returned depend on the chosen **mapping** and **optimizer**. This dictionary can be directly pass to
        the **optimize_options** argument of the optimize method `optimize` (or `Model.optimize`).

    See Also
    --------
    smash.optimize : Model assimilation using numerical optimization algorithms.

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

    Directly pass this dictionary to the **optimize_options** argument of the optimize method
    `optimize` (or `Model.optimize`).
    It's equivalent to set **optimize_options** to None (which is the default value)

    >>> model_u = smash.optimize(model, mapping="uniform", optimize_options=opt_u)
    </> Optimize
        At iterate      0    nfg =     1    J =      0.695010    ddx = 0.64
        At iterate      1    nfg =    30    J =      0.098411    ddx = 0.64
        At iterate      2    nfg =    59    J =      0.045409    ddx = 0.32
        At iterate      3    nfg =    88    J =      0.038182    ddx = 0.16
        At iterate      4    nfg =   117    J =      0.037362    ddx = 0.08
        At iterate      5    nfg =   150    J =      0.037087    ddx = 0.02
        At iterate      6    nfg =   183    J =      0.036800    ddx = 0.02
        At iterate      7    nfg =   216    J =      0.036763    ddx = 0.01
        CONVERGENCE: DDX < 0.01

    Customize the optimize options dictionary by removing ``'kexc'`` from the optimized parameters

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

    Run the optimize method

    >>> model_u = smash.optimize(model, mapping="uniform", optimize_options=opt_u)
    ValueError: Unknown or non optimized parameter 'kexc' in bounds optimize_options.
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

    Run again the optimize method to see the differences linked to a change in control vector

    >>> model_u = smash.optimize(model, mapping="uniform", optimize_options=opt_u)
    </> Optimize
        At iterate      0    nfg =     1    J =      0.695010    ddx = 0.64
        At iterate      1    nfg =    17    J =      0.128863    ddx = 0.64
        At iterate      2    nfg =    32    J =      0.069483    ddx = 0.32
        At iterate      3    nfg =    49    J =      0.045072    ddx = 0.16
        At iterate      4    nfg =    65    J =      0.044047    ddx = 0.08
        At iterate      5    nfg =    84    J =      0.043528    ddx = 0.04
        At iterate      6    nfg =   102    J =      0.042690    ddx = 0.02
        At iterate      7    nfg =   122    J =      0.042665    ddx = 0.01
        At iterate      8    nfg =   140    J =      0.042606    ddx = 0.00
        CONVERGENCE: DDX < 0.0

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
        'learning_rate': 0.001,
        'random_state': None,
        'termination_crit': {'maxiter': 200, 'early_stopping': 0}
    }

    Again, customize the optimize options and optimize the Model

    >>> opt_ann["learning_rate"] = 0.006
    >>> opt_ann["termination_crit"]["maxiter"] = 50
    >>> opt_ann["termination_crit"]["early_stopping"] = 5
    >>> opt_ann["random_state"] = 21
    >>> model.optimize(mapping="ann", optimize_options=opt_ann)
    </> Optimize
        At iterate      0    nfg =    1    J =       1.222062    |proj g| =       0.000209
        At iterate      1    nfg =    2    J =       1.219307    |proj g| =       0.000240
        ...
        At iterate     40    nfg =   41    J =       0.052151    |proj g| =       0.013186
        At iterate     41    nfg =   42    J =       0.051206    |proj g| =       0.003747
        At iterate     42    nfg =   43    J =       0.057921    |proj g| =       0.005087
        At iterate     43    nfg =   44    J =       0.063805    |proj g| =       0.010100
        At iterate     44    nfg =   45    J =       0.065734    |proj g| =       0.013365
        At iterate     45    nfg =   46    J =       0.064539    |proj g| =       0.015616
        At iterate     46    nfg =   47    J =       0.063309    |proj g| =       0.017270
        EARLY STOPPING: NO IMPROVEMENT for 5 CONSECUTIVE ITERATIONS
        Reverting to iteration 41 with J = 0.051206 due to early stopping

    The training process was terminated after 46 iterations, where the loss did not decrease below the minimal
    value at iteration 41 for 5 consecutive iterations. The optimal parameters are thus recorded at epoch 41.
    """
    mapping, optimizer = _standardize_default_optimize_options_args(mapping, optimizer)

    return _standardize_simulation_optimize_options(model, "optimize", mapping, optimizer, None)


def default_bayesian_optimize_options(
    model: Model,
    mapping: str = "uniform",
    optimizer: str | None = None,
) -> dict:
    """
    Default bayesian optimize options of Model.

    Parameters
    ----------
    model : `Model`
        Primary data structure of the hydrological model `smash`.

    mapping : `str`, default 'uniform'
        Type of mapping. Should be one of

        - ``'uniform'``
        - ``'distributed'``
        - ``'multi-linear'``
        - ``'multi-polynomial'``

        .. hint::
            See the :ref:`math_num_documentation.mapping` section

    optimizer : `str` or None, default None
        Name of optimizer. Should be one of

        - ``'sbs'`` (``'uniform'`` **mapping** only)
        - ``'lbfgsb'`` (``'uniform'``, ``'distributed'``, ``'multi-linear'`` or ``'multi-polynomial'``
          **mapping** only)

        .. note::
            If not given, a default optimizer will be set depending on the optimization mapping:

            - **mapping** = ``'uniform'``; **optimizer** = ``'sbs'``
            - **mapping** = ``'distributed'``, ``'multi-linear'``, or ``'multi-polynomial'``; **optimizer** =
              ``'lbfgsb'``

        .. hint::
            See the :ref:`math_num_documentation.optimization_algorithm` section

    Returns
    -------
    optimize_options : `dict[str, Any]`
        Dictionary containing optimization options for fine-tuning the optimization process. The specific keys
        returned depend on the chosen **mapping** and **optimizer**. This dictionary can be directly pass to
        the **optimize_options** argument of the optimize method `bayesian_optimize` (or
        `Model.bayesian_optimize`).

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

    Directly pass this dictionary to the **optimize_options** argument of the optimize method
    `bayesian_optimize` (or `Model.bayesian_optimize`).
    It's equivalent to set **optimize_options** to None (which is the default value)

    >>> model_u = smash.bayesian_optimize(model, mapping="uniform", optimize_options=opt_u)
    </> Bayesian Optimize
        At iterate      0    nfg =     1    J =     77.049133    ddx = 0.64
        At iterate      1    nfg =    68    J =      2.584603    ddx = 0.64
        At iterate      2    nfg =   135    J =      2.324317    ddx = 0.32
        At iterate      3    nfg =   202    J =      2.304130    ddx = 0.08
        At iterate      4    nfg =   269    J =      2.262191    ddx = 0.08
        At iterate      5    nfg =   343    J =      2.260251    ddx = 0.01
        At iterate      6    nfg =   416    J =      2.258220    ddx = 0.00
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

    Customize the bayesian optimize options and optimize the Model

    >>> opt_ml["bounds"]["cp"] = (1, 2000)
    >>> opt_ml["bounds"]["sg0"] = (1e-3, 100)
    >>> opt_ml["descriptor"]["cp"] = "slope"
    >>> opt_ml["termination_crit"]["maxiter"] = 10
    >>> model.bayesian_optimize(mapping="multi-linear", optimize_options=opt_ml)
    </> Bayesian Optimize
        At iterate      0    nfg =     1    J =     77.049095    |proj g| =    147.958771
        At iterate      1    nfg =     2    J =      6.694370    |proj g| =      4.301311
        At iterate      2    nfg =     3    J =      6.527157    |proj g| =      3.754591
        At iterate      3    nfg =     4    J =      5.088758    |proj g| =      2.261085
        At iterate      4    nfg =     5    J =      4.736641    |proj g| =      1.637916
        At iterate      5    nfg =     6    J =      4.421250    |proj g| =      1.300914
        At iterate      6    nfg =     7    J =      4.284939    |proj g| =      1.345769
        At iterate      7    nfg =     8    J =      4.196455    |proj g| =      1.179024
        At iterate      8    nfg =     9    J =      4.139528    |proj g| =      1.159659
        At iterate      9    nfg =    10    J =      4.099973    |proj g| =      1.042880
        At iterate     10    nfg =    11    J =      4.027408    |proj g| =      1.142047
        STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT

    The optimization process was terminated after 10 iterations, the maximal value we defined.
    """

    mapping, optimizer = _standardize_default_bayesian_optimize_options_args(mapping, optimizer)

    return _standardize_simulation_optimize_options(model, "bayesian_optimize", mapping, optimizer, None)
