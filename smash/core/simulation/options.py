from __future__ import annotations

from smash.core.simulation._standardize import (
    _standardize_simulation_optimize_options,
    _standardize_default_optimize_options_args,
    _standardize_default_bayesian_optimize_options_args,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model

__all__ = ["default_optimize_options", "default_bayesian_optimize_options"]


def default_optimize_options(
    model: Model,
    mapping: str = "uniform",
    optimizer: str | None = None,
) -> dict:
    """
    Get the default optimization options for the Model object.

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

    Returns
    -------
    optimize_options : dict
        A dictionary containing optimization options for optimizing the Model object. The specific keys returned depend on the chosen mapping and optimizer. All possible keys are:

        parameters : ListLike
            Operator parameter(s) and/or initial state(s) to be optimized. It must consist of the parameters defined in the Model setup.

        bounds : dict
            Bounds on the optimized parameters. This is a dictionary where the keys represent parameter and/or state names, and the values are pairs of ``(min, max)`` values (i.e., a list or tuple) with ``min`` lower than ``max``.

        control_tfm : str
            Transformation methods applied to the control vector: 'keep', 'normalize', 'sbs'. Only used when **optimizer** is 'sbs' or 'lbfgsb'.

        descriptor : dict
            A dictionary containing lists of descriptors used for each operator parameter.
            Only used when **mapping** is 'multi-linear' or 'multi-polynomial'. In case of 'ann', all descriptors will be used.

        net : Net
            The neural network used to learn the descriptors-to-parameters mapping. Only used when **mapping** is 'ann'.

            .. hint::
                Refer to `smash.factory.Net` to learn how to create a customized neural network for training.

        learning_rate : float
            The learning rate used for weight updates during training. Only used when **mapping** is 'ann'.

        random_state : int
            A random seed used to initialize weights. A value of None indicates that weights will be initialized with a random seed. Only used when **mapping** is 'ann'.

        termination_crit : dict
            Termination criteria. The keys are:

            - 'maxiter': The maximum number of iterations. Only used when **optimizer** is 'sbs' or 'lbfgsb'.
            - 'factr': An additional termination criterion based on cost values. Only used when **optimizer** is 'lbfgsb'.
            - 'pgtol': An additional termination criterion based on the projected gradient of the cost function. Only used when **optimizer** is 'lbfgsb'.
            - 'epochs': The number of training epochs for the neural network. Only used when **mapping** is 'ann'.
            - 'early_stopping': A positive number to stop training when the loss function does not decrease below the current optimal value for **early_stopping** consecutive epochs. When set to zero, early stopping is disabled, and the training continues for the full number of epochs. Only used when **mapping** is 'ann'.

    Examples
    --------
    >>> import smash
    >>> from smash.factory import load_dataset
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Get the default optimiaztion options for multi-linear mapping:

    >>> opt_ml = smash.default_optimize_options(model, mapping="multi-linear")
    >>> opt_ml
    {
        'parameters': ['cp', 'ct', 'kexc', 'llr'],
        'bounds': {
                    'cp': (1e-06, 1000.0), 'ct': (1e-06, 1000.0),
                    'kexc': (-50, 50), 'llr': (1e-06, 1000.0)
                  },
        'control_tfm': 'normalize',
        'descriptor': {
                        'cp': array(['slope', 'dd'], dtype='<U5'),
                        'ct': array(['slope', 'dd'], dtype='<U5'),
                        'kexc': array(['slope', 'dd'], dtype='<U5'),
                        'llr': array(['slope', 'dd'], dtype='<U5')
                      },
        'termination_crit': {'maxiter': 100, 'factr': 1000000.0, 'pgtol': 1e-12}
    }

    For ANN-based mapping:

    >>> opt_ann = smash.default_optimize_options(model, mapping="ann")
    >>> opt_ann
    {
        'parameters': ['cp', 'ct', 'kexc', 'llr'],
        'bounds': {
                    'cp': (1e-06, 1000.0), 'ct': (1e-06, 1000.0),
                    'kexc': (-50, 50), 'llr': (1e-06, 1000.0)
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
        'termination_crit': {'epochs': 200, 'early_stopping': 0}
    }

    Customize the optimization options and optimize the Model:

    >>> opt_ann["learning_rate"] = 0.004
    >>> opt_ann["termination_crit"]["epochs"] = 40
    >>> opt_ann["termination_crit"]["early_stopping"] = 10
    >>> opt_ann["random_state"] = 11
    >>> model.optimize(mapping="ann", optimize_options=opt_ann)
    </> Optimize
        At epoch      1    J =  1.150821    |proj g| =  0.000060
        At epoch      2    J =  1.142061    |proj g| =  0.000067
        ...
        At epoch     26    J =  0.103324    |proj g| =  0.000159
        At epoch     27    J =  0.089769    |proj g| =  0.000235
        At epoch     28    J =  0.104150    |proj g| =  0.000305
        ...
        At epoch     36    J =  0.185123    |proj g| =  0.000176
        At epoch     37    J =  0.179911    |proj g| =  0.000177
        Training:  92%|██████████████████████████▊  | 37/40 [00:30<00:02,  1.23it/s]

    The training process was terminated after 37 epochs, where the loss did not decrease below the minimal value at epoch 27 for 10 consecutive epochs.
    The optimal parameters are thus recorded at epoch 27:

    >>> metric = smash.metrics(model)
    >>> 1 - metric[0]  # cost at downstream gauge, which was calibrated
    0.08976912498474121
    """

    mapping, optimizer = _standardize_default_optimize_options_args(mapping, optimizer)

    return _standardize_simulation_optimize_options(
        model, "optimize", mapping, optimizer, None
    )


def default_bayesian_optimize_options(
    model: Model,
    mapping: str = "uniform",
    optimizer: str | None = None,
) -> dict:
    """
    Get the default bayesian optimization options for the Model object.

    Parameters
    ----------
    model : Model
        Model object.

    mapping : str, default 'uniform'
        Type of mapping. Should be one of 'uniform', 'distributed', 'multi-linear', 'multi-polynomial'.

    optimizer : str or None, default None
        Name of optimizer. Should be one of 'sbs', 'lbfgsb'.

        .. note::
            If not given, a default optimizer will be set depending on the optimization mapping:

            - **mapping** = 'uniform'; **optimizer** = 'sbs'
            - **mapping** = 'distributed', 'multi-linear', or 'multi-polynomial'; **optimizer** = 'lbfgsb'

    Returns
    -------
    optimize_options : dict
        A dictionary containing optimization options for optimizing the Model object. The specific keys returned depend on the chosen mapping and optimizer. All possible keys are:

        parameters : ListLike
            Operator parameter(s) and/or initial state(s) to be optimized. It must consist of the parameters defined in the Model setup.

        bounds : dict
            Bounds on the optimized parameters. This is a dictionary where the keys represent parameter and/or state names, and the values are pairs of ``(min, max)`` values (i.e., a list or tuple) with ``min`` lower than ``max``.

        control_tfm : str
            Transformation methods applied to the control vector: 'keep', 'normalize', 'sbs'. Only used when **optimizer** is 'sbs' or 'lbfgsb'.

        descriptor : dict
            A dictionary containing lists of descriptors used for each operator parameter.
            Only used when **mapping** is 'multi-linear' or 'multi-polynomial'.

        termination_crit : dict
            Termination criteria. The keys are:

            - 'maxiter': The maximum number of iterations. Only used when **optimizer** is 'sbs' or 'lbfgsb'.
            - 'factr': An additional termination criterion based on cost values. Only used when **optimizer** is 'lbfgsb'.
            - 'pgtol': An additional termination criterion based on the projected gradient of the cost function. Only used when **optimizer** is 'lbfgsb'.

    Examples
    --------
    >>> import smash
    >>> from smash.factory import load_dataset
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Get the default bayesian optimiaztion options for multi-linear mapping:

    >>> bay_opt_ml = smash.default_bayesian_optimize_options(model, mapping="multi-linear")
    >>> bay_opt_ml
        {
            'parameters': ['cp', 'ct', 'kexc', 'llr', 'sg0', 'sg1'],
            'bounds': {
                        'cp': (1e-06, 1000.0), 'ct': (1e-06, 1000.0),
                        'kexc': (-50, 50), 'llr': (1e-06, 1000.0),
                        'sg0': (1e-06, 1000.0), 'sg1': (1e-06, 10.0)
                      },
            'control_tfm': 'normalize',
            'descriptor': {
                            'cp': array(['slope', 'dd'], dtype='<U5'),
                            'ct': array(['slope', 'dd'], dtype='<U5'),
                            'kexc': array(['slope', 'dd'], dtype='<U5'),
                            'llr': array(['slope', 'dd'], dtype='<U5'),
                            'sg0': array(['slope', 'dd'], dtype='<U5'),
                            'sg1': array(['slope', 'dd'], dtype='<U5')
                          },
            'termination_crit': {'maxiter': 100, 'factr': 1000000.0, 'pgtol': 1e-12}}
    """

    mapping, optimizer = _standardize_default_bayesian_optimize_options_args(
        mapping, optimizer
    )

    return _standardize_simulation_optimize_options(
        model, "bayesian_optimize", mapping, optimizer, None
    )
