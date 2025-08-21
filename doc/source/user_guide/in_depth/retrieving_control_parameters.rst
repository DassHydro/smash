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
    ...
    >>> # Calibrate model and store intermediate results using callback
    >>> model_u = smash.optimize(model, callback=callback_func, optimize_options={"termination_crit": {"maxiter": 5}})

.. code-block:: output

    </> Optimize
        At iterate     0    nfg =     1    J = 6.85771e-01    ddx = 0.64
        Callback message: stored control values at iteration 1 in control_vector_iter_1.txt
        At iterate     1    nfg =    30    J = 3.51670e-01    ddx = 0.64
        Callback message: stored control values at iteration 2 in control_vector_iter_2.txt
        At iterate     2    nfg =    58    J = 1.80573e-01    ddx = 0.32
        Callback message: stored control values at iteration 3 in control_vector_iter_3.txt
        At iterate     3    nfg =    88    J = 1.77831e-01    ddx = 0.16
        Callback message: stored control values at iteration 4 in control_vector_iter_4.txt
        At iterate     4    nfg =   117    J = 1.74783e-01    ddx = 0.08
        Callback message: stored control values at iteration 5 in control_vector_iter_5.txt
        At iterate     5    nfg =   149    J = 1.69981e-01    ddx = 0.08
        STOP: TOTAL NO. of ITERATIONS REACHED LIMIT

The callback function in the example above stores the control vector at each iteration of the optimization process. 
For instance, we can read the control values at iteration 5:

.. code-block:: python

    >>> control_vector_u = np.loadtxt("control_vector_iter_5.txt")
    >>> control_vector_u

.. code-block:: output

    array([4.41831732, 4.69460869, 1.12      , 6.14749146])

Control vector information
--------------------------

Now, for better understanding of the values of the control vector, we can use the `smash.optimize_control_info` method to get the information on control vector for the current optimization configuration (corresponding function `smash.bayesian_optimize_control_info` if using `smash.bayesian_optimize`).

.. code-block:: python

    >>> control_info = smash.optimize_control_info(model_u)

.. note::
    The optional arguments in the `smash.optimize_control_info` function (such as ``mapping``, ``optimizer``, and relevant arguments in ``optimize_options``) must match those used or will be used in the calibration process with `smash.optimize` to ensure consistency in the optimization problem setup.
    The same applies to `smash.bayesian_optimize_control_info` when using `smash.bayesian_optimize`.
    In the example above, we used the default values for these arguments.

The ``control_info`` dictionary provides detailed information about the optimization control parameters, such as the size, names, values, and bounds of the control vector.
For instance, the previously optimized control values can be accessed as follows:

.. code-block:: python

    >>> control_info["x"]

.. code-block:: output

    array([4.4183197, 4.6946096, 1.1200012, 6.147493 ], dtype=float32)

These values are the same as those in ``control_vector_u`` and correspond to the following parameters:

.. code-block:: python

    >>> control_info["name"].tolist()

.. code-block:: output

    ['cp-0', 'ct-0', 'kexc-0', 'llr-0']

Here, the names are in the format ``<key>-0``, where ``<key>`` represents the conceptual model parameters/states, and ``0`` indicates that these parameters/states are spatially uniform since we used the default mapping ``mapping='uniform'``.
For instance, the optimized value:

.. code-block:: python

    >>> control_vector_u.tolist()[1]

.. code-block:: output

    4.694608688354492

corresponds to the parameter:

.. code-block:: python

    >>> control_info["name"][1].split("-")[0]

.. code-block:: output

    'ct'

which is spatially uniform.

.. note::
    The control values may differ from the actual conceptual parameters/states because transformation functions might be applied to the control vector before the optimization process 
    (refer to the documentation of ``control_tfm`` in the ``optimize_options`` argument of `smash.optimize` or `smash.bayesian_optimize` for more information).
    The control values without any transformation applied are:

    .. code-block:: python

        >>> control_info["x_raw"]  # raw values before transformation

    .. code-block:: output
        
        array([ 82.95676  , 109.3561   ,   1.3692893, 467.5437   ], dtype=float32)

Retrieving control parameters and continuing model calibration
--------------------------------------------------------------

Next, we will retrieve the control parameters from the stored control vectors and continue the calibration process.
This tutorial provides an example using a uniform mapping, but the same approach can be applied to more complex mappings for higher-dimensional optimization, such as using a distributed mapping or neural networks for regionalization.

Set the control vector values to the model using `Model.set_control_optimize <smash.Model.set_control_optimize>` (corresponding function `Model.set_control_bayesian_optimize <smash.Model.set_control_bayesian_optimize>` if using `smash.bayesian_optimize`).

.. code-block:: python

    >>> model_u.set_control_optimize(control_vector_u)

.. note::
    The optional arguments in the `Model.set_control_optimize <smash.Model.set_control_optimize>` method (such as ``mapping``, ``optimizer``, and relevant arguments in ``optimize_options``) must match those used in the previous calibration with `smash.optimize` to ensure consistency in the optimization problem setup.
    The same applies to `Model.set_control_bayesian_optimize <smash.Model.set_control_bayesian_optimize>` when using `smash.bayesian_optimize`.
    In the example above, we used the default values for these arguments.

Finally, we can continue the calibration using the starting point defined by the control vector obtained from the previous calibration:

.. code-block:: python

    >>> model_u.optimize()

.. code-block:: output

    </> Optimize
        At iterate     0    nfg =     1    J = 1.69981e-01    ddx = 0.64
        At iterate     1    nfg =    32    J = 1.69542e-01    ddx = 0.08
        At iterate     2    nfg =    61    J = 1.64806e-01    ddx = 0.08
        At iterate     3    nfg =    89    J = 1.61149e-01    ddx = 0.08
        At iterate     4    nfg =   117    J = 1.56910e-01    ddx = 0.08
        At iterate     5    nfg =   151    J = 1.55379e-01    ddx = 0.04
        At iterate     6    nfg =   187    J = 1.52012e-01    ddx = 0.04
        At iterate     7    nfg =   222    J = 1.51865e-01    ddx = 0.02
        At iterate     8    nfg =   256    J = 1.51631e-01    ddx = 0.01
        At iterate     9    nfg =   290    J = 1.51580e-01    ddx = 0.01
        At iterate    10    nfg =   306    J = 1.51580e-01    ddx = 0.01
        CONVERGENCE: DDX < 0.01

As observed, the optimization process resumes from the previously interrupted point, with the initial cost value being the same as the one at iteration 5 from the previous calibration.

.. warning::
    Due to technical limitations, the continuous calibration process currently does not work with ``mapping='multi-linear'`` or ``mapping='multi-power'``.
    However, you can still set the control values to the model to retrieve the model parameters and then perform a forward run to update the hydrological responses and final states.

    .. code-block:: python

        >>> # Define the optimize_options. Here we use the default values.
        >>> optimize_options_ml = smash.default_optimize_options(model, mapping="multi-linear")
        >>> # Example of a control vector obtained from calibration with multi-linear mapping:
        >>> # model.optimize(mapping="multi-linear", optimize_options=optimize_options_ml)
        >>> control_vector_ml = np.array([
        ...     -1.94, -0.07, -0.32, -0.66, -0.17, 0.29, 0.11, -0.67, -0.04, -0.33,
        ...     -0.62, -0.15, 0.24, 0.04, 0, 0.11, 0, 0.29, 0.15, -0.66, 
        ...     -0.43, -1.97, 0, 0.01, 0.02, 0, 0.01, 0.01
        ... ], dtype=np.float32)
        >>> # Retrieve model parameters using control_vector_ml
        >>> model.set_control_optimize(control_vector_ml, mapping="multi-linear", optimize_options=optimize_options_ml)
        >>> model.forward_run()  # update hydrological responses and final states
        >>> # Check the cost value (1-NSE) of the most downstream gauge (which is the default calibrated gauge)
        >>> print("    Cost value (1-NSE) =", 1 - smash.evaluation(model)[0][0])

    .. code-block:: output

        </> Forward Run
            Cost value (1-NSE) = 0.30294644832611084

    Continuous calibration currently does not work with multiple linear and multiple power mappings.
    However, for multiple linear mapping, an alternative solution to continue the calibration process from an obtained control vector is to create a neural network without hidden layers (equivalent to multiple linear regression) and set the obtained control vector as the initial weights of the neural network. 
    You can then continue to calibrate the model with ``mapping='ann'``.

    .. code-block:: python

        >>> def set_control_to_ann(control_vector: np.ndarray, bound_parameters: dict) -> smash.factory.Net:
        ...     # Define number of descriptors/parameters
        ...     n_descriptors = 6  # (to be adapted based on your config)
        ...     n_parameters = len(bound_parameters)
        ...     # Preprocess control vector based on control information
        ...     rs_control = control_vector.reshape(n_parameters, n_descriptors + 1)
        ...     # Create a neural network without hidden layers
        ...     net = smash.factory.Net()
        ...     net.add_dense(n_parameters, input_shape=n_descriptors, activation="sigmoid")
        ...     net.add_scale(list(bound_parameters.values()))
        ...     # Set the control vector to the neural network weights
        ...     net.set_weight([rs_control[:, 1:]])
        ...     net.set_bias([rs_control[:, 0].reshape(1, -1)])
        ...     return net
        ...
        >>> net = set_control_to_ann(control_vector_ml, optimize_options_ml["bounds"])
        >>> # Retrieve optimize_options from optimize_options_ml
        >>> optimize_options_ann = {"parameters": optimize_options_ml["parameters"], "bounds": optimize_options_ml["bounds"]}
        >>> # Define optimize_options in case of ANN
        >>> optimize_options_ann["net"] = net
        >>> optimize_options_ann["learning_rate"] = 0.003
        >>> optimize_options_ann["termination_crit"] = {"maxiter": 5}
        >>> # Continue calibration with ANN
        >>> model.optimize(mapping="ann", optimize_options=optimize_options_ann)

    .. code-block:: output

        </> Optimize
            At iterate     0    nfg =     1    J = 3.02946e-01    |proj g| = 1.64296e-03
            At iterate     1    nfg =     2    J = 3.01436e-01    |proj g| = 1.05929e-03
            At iterate     2    nfg =     3    J = 3.00575e-01    |proj g| = 5.02047e-04
            At iterate     3    nfg =     4    J = 3.00441e-01    |proj g| = 7.04148e-04
            At iterate     4    nfg =     5    J = 3.00257e-01    |proj g| = 9.07534e-04
            At iterate     5    nfg =     6    J = 2.99827e-01    |proj g| = 8.86482e-04
            STOP: TOTAL NO. of ITERATIONS REACHED LIMIT

    We observe that the initial cost value is the same as the one obtained in the case of multi-linear mapping.

.. use this directive to hide the code cell in the documentation while being captured by the script pyexec_rst.py
.. only:: never

    .. code-block:: python

        >>> import os
        >>> import glob
        >>> for file in glob.glob("control_vector_iter_*.txt"):
        ...     os.remove(file)