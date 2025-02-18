.. _user_guide.in_depth.retrieving_control_parameters:

========================================================
Retrieving Control Parameters for Continuous Calibration
========================================================

Calibrating hydrological models can be a lengthy process, especially when dealing with large models and datasets.
For various reasons (such as interruptions during optimization or the need to re-run the calibration), it is beneficial to retrieve the control parameters of a previously calibrated model.

This tutorial demonstrates how to retrieve the control parameters of a model that has been calibrated using either `smash.optimize` or `smash.bayesian_optimize`.
For simplicity, this guide will focus on the `smash.optimize` case.

We begin by opening a Python interface:

.. code-block:: none

    python3

Imports
-------

We will first import the necessary libraries for this tutorial.

.. code-block:: python

    >>> import smash
    >>> import numpy as np

Model creation
--------------

Now, we need to create a :class:`smash.Model` object.
For this case, we will use the :ref:`user_guide.data_and_format_description.lez` dataset as an example.

Load the ``setup`` and ``mesh`` dictionaries using the `smash.factory.load_dataset` function and create the :class:`smash.Model` object.

.. code-block:: python

    >>> setup, mesh = smash.factory.load_dataset("Lez")
    >>> model = smash.Model(setup, mesh)

Model calibration with callback function
----------------------------------------

To store intermediate results (e.g., cost values, control vector, etc.) of the calibration process, we recommend using a callback function.
A callback function can be passed to the `smash.optimize` method (and  `smash.bayesian_optimize` method) to capture these results at each iteration of the optimization process.

Here is an example of how to define and use a callback function in a calibration process:

.. code-block:: python

    >>> # Define callback function
    >>> def callback_func(iopt):
    ...     file_name = f"control_vector_iter_{iopt.n_iter}.txt"
    ...     # Store the control vector at the current optimization iteration
    ...     np.savetxt(file_name, iopt.control_vector)
    ...     print(f"    Callback message: stored control values at iteration {iopt.n_iter} in {file_name}")
    >>> 
    >>> # Calibrate model and store intermediate results using callback
    >>> model.optimize(callback=callback_func)

.. code-block:: output

    </> Optimize
        At iterate     0    nfg =     1    J = 6.85771e-01    ddx = 0.64
        Callback message: stored control values at iteration 1 in control_vector_iter_1.txt
        At iterate     1    nfg =    30    J = 3.51670e-01    ddx = 0.64
        ...
        At iterate     9    nfg =   282    J = 1.52578e-01    ddx = 0.08
        Callback message: stored control values at iteration 10 in control_vector_iter_10.txt
        At iterate    10    nfg =   317    J = 1.51915e-01    ddx = 0.04
        Callback message: stored control values at iteration 11 in control_vector_iter_11.txt
        At iterate    11    nfg =   350    J = 1.51743e-01    ddx = 0.02
        Callback message: stored control values at iteration 12 in control_vector_iter_12.txt
        At iterate    12    nfg =   383    J = 1.51628e-01    ddx = 0.01
        Callback message: stored control values at iteration 13 in control_vector_iter_13.txt
        At iterate    13    nfg =   407    J = 1.51613e-01    ddx = 0.01
        CONVERGENCE: DDX < 0.01

The callback function in the example above stores the control vector at each iteration of the optimization process. 
For instance, we can read the control values at iteration 9:

.. code-block:: python

    >>> control_vector = np.loadtxt("control_vector_iter_9.txt")
    >>> control_vector

.. code-block:: output

    array([5.06575155, 4.00118542, 0.91746539, 6.21497154])

Control vector information
--------------------------

Now, for better understanding of the values of the control vector, we can use the `smash.optimize_control_info` method to get the information on control vector for the current optimization configuration (corresponding function `smash.bayesian_optimize_control_info` if using `smash.bayesian_optimize`).

.. code-block:: python

    >>> control_info = smash.optimize_control_info(model)

.. note::
    All optional arguments in the `smash.optimize_control_info` function (such as ``mapping``, ``optimizer``, ``optimize_options``) define the optimization configuration used or will be used during the calibration process. 
    Therefore, these arguments must match the ones used or intended to be used in `smash.optimize` (similarly for `smash.bayesian_optimize_control_info` if using `smash.bayesian_optimize`).
    In the example above, we used the default values for these arguments.

The ``control_info`` dictionary contains detailed information about the optimization control parameters, such as the size, names, values, and bounds of the control vector.
For example, the values in the ``control_vector`` above correspond to the following parameters:

.. code-block:: python

    >>> control_info["name"]

.. code-block:: output

    array(['cp-0', 'ct-0', 'kexc-0', 'llr-0'], dtype='<U128')

Here, the names are in the format ``<key>-0``, where ``<key>`` represents the conceptual model parameters, and ``0`` indicates that these parameters are spatially uniform since we used the default mapping ``mapping='uniform'``.
For instance, the optimized value:

.. code-block:: python

    >>> control_vector[1]

.. code-block:: output

    np.float64(4.001185417175293)

corresponds to the parameter:

.. code-block:: python

    >>> control_info["name"][1].split("-")[0]

.. code-block:: output

    'ct'

which is spatially uniform.

.. note::
    The control values may differ from the actual conceptual parameters because transformation functions might be applied to the control vector before the optimization process.
    Refer to the documentation of ``control_tfm`` in the ``optimize_options`` argument of `smash.optimize` (or `smash.bayesian_optimize`) for more information.

Retrieving Control Parameters
-----------------------------

Next, we will retrieve the control parameters from the stored control vectors and continue the calibration process.
This tutorial provides an example using a uniform mapping, but the same approach can be applied to more complex mappings for higher-dimensional optimization, such as using a distributed mapping or neural networks for regionalization.

Set the control vector values to the model using `Model.set_control_optimize <smash.Model.set_control_optimize>` (corresponding function `Model.set_control_bayesian_optimize <smash.Model.set_control_bayesian_optimize>` if using `smash.bayesian_optimize`).

.. code-block:: python

    >>> model.set_control_optimize(control_vector)

.. note::
    All optional arguments in the `Model.set_control_optimize <smash.Model.set_control_optimize>` method (such as ``mapping``, ``optimizer``, ``optimize_options``) define the optimization configuration used during the previous calibration. 
    Therefore, these arguments must match the ones used in `smash.optimize` (similarly for `Model.set_control_bayesian_optimize <smash.Model.set_control_bayesian_optimize>` if using `smash.bayesian_optimize`).
    In the example above, we used the default values for these arguments.

Finally, we can continue the calibration using the starting point defined by the control vector obtained from the previous calibration:

.. code-block:: python

    >>> model.optimize()

.. code-block:: output

    </> Optimize
        At iterate     0    nfg =     1    J = 1.52578e-01    ddx = 0.64
        At iterate     1    nfg =    33    J = 1.52578e-01    ddx = 0.04
        At iterate     2    nfg =    62    J = 1.51770e-01    ddx = 0.04
        At iterate     3    nfg =    91    J = 1.51607e-01    ddx = 0.01
        At iterate     4    nfg =   106    J = 1.51595e-01    ddx = 0.01
        CONVERGENCE: DDX < 0.01

.. warning::
    Due to technical limitations, the continuous calibration process does not work with ``mapping='multi-linear'`` or ``mapping='multi-polynomial'``.
    However, if you are using a multiple linear mapping, an alternative solution to continue the calibration process from an obtained control vector is to create a neural network without hidden layers (equivalent to multiple linear regression) and set the obtained control vector as the initial weights of the neural network. 
    You can then continue to calibrate the model with ``mapping='ann'``.

    .. use pycon to preserve Python format and avoid being captured by script pyexec_rst.py

    .. code-block:: pycon

        >>> def set_control_to_ann(control_vector: np.ndarray, model: smash.Model) -> smash.factory.Net:
        ...     # Initialize a neural network
        ...     net = smash.factory.Net()
        ... 
        ...     # Add layers to create a neural network without hidden layers
        ...     # using smash.factory.Net methods and model
        ...     ...
        ... 
        ...     # Set the control vector to the neural network weights
        ...     # using net.set_weight and net.set_bias methods
        ...     ...
        ... 
        ...     return net
        >>> net = set_control_to_ann(control_vector, model)
        >>> 
        >>> model.optimize(mapping="ann", optimize_options={"net": net})

.. only:: never

    .. code-block:: python

        >>> import os
        >>> import glob
        >>> for file in glob.glob("control_vector_iter_*.txt"):
        ...     os.remove(file)