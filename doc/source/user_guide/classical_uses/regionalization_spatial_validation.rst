.. _user_guide.classical_uses.regionalization_spatial_validation:

======================================
Regionalization and Spatial Validation
======================================

This tutorial explains how to perform regionalization and spatial validation methods with `smash` using physical descriptors.
The parameters :math:`\boldsymbol{\theta}` can be written as a mapping :math:`\phi` of descriptors :math:`\boldsymbol{\mathcal{D}}`
(slope, drainage density, soil water storage, etc) and :math:`\boldsymbol{\rho}` a control vector:
:math:`\boldsymbol{\theta}(x)=\phi\left(\boldsymbol{\mathcal{D}}(x),\boldsymbol{\rho}\right)`.
See the :ref:`math_num_documentation.mapping` section for more details.

First, a shape is assumed for the mapping (here **multi-polynomial** or **neural network**).
Then the control vector of the mapping needs to be optimized: :math:`\boldsymbol{\hat{\rho}}=\underset{\mathrm{\boldsymbol{\rho}}}{\text{argmin}}\;J`,
with :math:`J` the cost function.

We begin by opening a Python interface:

.. code-block:: none

    python3

.. ipython:: python
    :suppress:

    import os

Imports
-------

We will first import everything we need in this tutorial.

.. ipython:: python

    import smash
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

Model Creation and Descriptors Visualization
--------------------------------------------

Now, we need to create a :class:`smash.Model` object.
For this case, we will use the :ref:`user_guide.demo_data.lez` dataset as an example.

Load the ``setup`` and ``mesh`` dictionaries using the `smash.factory.load_dataset` function and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.factory.load_dataset("Lez")
    model = smash.Model(setup, mesh)

Six physical descriptors are considered in this example, which are:

.. image:: ../../_static/physio_descriptors.png
    :align: center

.. TODO: Add descriptor explanation

The values of these descriptors can be obtained in the ``physio_data`` derived type of the :class:`smash.Model` object.

.. ipython:: python

    model.physio_data.descriptor.shape  # (x, y, n_descriptors)

Model simulation
----------------

Multiple polynomial
*******************

To optimize the rainfall-runoff model using a multiple polynomial mapping of descriptors to conceptual model parameters,
the value ``multi-polynomial`` simply needs to be passed to the mapping argument. We add another option to limit the number of iterations
by stopping the optimizer after ``50`` iterations.

.. To speed up documentation generation
.. ipython:: python
    :suppress:

    ncpu = min(5, max(1, os.cpu_count() - 1))
    model_mp = smash.optimize(
        model,
        mapping="multi-polynomial",
        optimize_options={
            "termination_crit": dict(maxiter=50),
        },
        common_options={"ncpu": ncpu},
    )

.. ipython:: python
    :verbatim:

    model_mp = smash.optimize(
        model,
        mapping="multi-polynomial",
        optimize_options={
            "termination_crit": dict(maxiter=50),
        },
    )

We have therefore optimized the set of rainfall-runoff parameters using a multiple polynomial regression constrained by
physiographic descriptors. Here, most of the options used are the default ones, i.e., a minimization of one minus the Nash-Sutcliffe
efficiency on the most downstream gauge of the domain. The resulting rainfall-runoff parameter maps can be viewed.

.. ipython:: python

    f, ax = plt.subplots(2, 2)

    map_cp = ax[0,0].imshow(model_mp.get_rr_parameters("cp"));
    f.colorbar(map_cp, ax=ax[0,0], label="cp (mm)");
    map_ct = ax[0,1].imshow(model_mp.get_rr_parameters("ct"));
    f.colorbar(map_ct, ax=ax[0,1], label="ct (mm)");
    map_kexc = ax[1,0].imshow(model_mp.get_rr_parameters("kexc"));
    f.colorbar(map_kexc, ax=ax[1,0], label="kexc (mm/d)");
    map_llr = ax[1,1].imshow(model_mp.get_rr_parameters("llr"));
    @savefig user_guide.classical_uses.regionalization_spatial_validation.mp_theta.png
    f.colorbar(map_llr, ax=ax[1,1], label="llr (min)");

As well as performances at upstream gauges

.. ipython:: python
    
    metrics = ["nse", "kge"]

    scores = np.round(smash.evaluation(model_mp, metrics)[1:, :], 2)

    upstream_perf = pd.DataFrame(data=scores, index=model.mesh.code[1:], columns=metrics)
    upstream_perf

.. note::
    The two upstream gauges are the two last gauges of the list. This is why we use ``[1:]`` in the lists in order to take all the gauges
    except the first, which is the downstream gauge on which the model has been calibrated.

Artificial neural network
*************************

We can optimize the rainfall-runoff model using a neural network (NN) based mapping of descriptors to conceptual model parameters.
It is possible to define your own network to implement this optimization, but here we willl use the default neural network.
Similar to multiple polynomial mapping, all you have to do is to pass the value, ``ann`` to the ``mapping`` argument.
We also pass other options specific to the use of a NN:

- ``optimize_options``
    - ``random_state``: a random seed used to initialize neural network weights.
    - ``learning_rate``: the learning rate used for weights updates during training.
    - ``termination_crit``: the maximum number of training ``maxiter`` for the neural network and a positive number to stop training when the loss function does not decrease below the current optimal value for ``early_stopping`` consecutive iterations.

- ``return_options``
    - ``net``: return the optimized neural network

.. To speed up documentation generation
.. ipython:: python
    :suppress:

    ncpu = min(5, max(1, os.cpu_count() - 1))
    model_ann, opt_ann = smash.optimize(
        model,
        mapping="ann",
        optimize_options={
            "random_state": 0,
            "learning_rate": 0.003,
            "termination_crit": dict(maxiter=80, early_stopping=20),
        },
        return_options={"net": True},
        common_options={"ncpu": ncpu},
    )

.. ipython:: python
    :verbatim:

    model_ann, opt_ann = smash.optimize(
        model,
        mapping="ann",
        optimize_options={
            "random_state": 0,
            "learning_rate": 0.003,
            "termination_crit": dict(maxiter=80, early_stopping=20),
        },
        return_options={"net": True},
    )
.. note::
    As we used the `smash.optimize` method (here an :ref:`ADAM algorithm <math_num_documentation.optimization_algorithm>` by default when choosing a NN based mapping) and asked for optional return values, this function will return two values, the optimized model
    ``model_ann`` and the optional returns ``opt_ann``.

.. hint::
    For advanced techniques, such as using customized ANNs, transfer learning, and more,
    refer to the in-depth tutorial on :ref:`Learnable Regionalization Mapping <user_guide.in_depth.advanced_learnable_regionalization>`.

Since we have returned the optimized neural network, we can visualize what it contains

.. ipython:: python

    opt_ann.net

The above information indicates that the default neural network is composed of 3 hidden dense layers, each followed by a ``ReLU`` activation function.
The output layer is followed by a ``TanH`` (hyperbolic tangent) function and it outputs in :math:`\left]-1,1\right[` are scaled to given conceptual parameter bounds using a ``MinMaxScale`` function.
Other information is available in the `smash.factory.Net` object, including the value of the cost function at each iteration.

.. ipython:: python

    plt.plot(opt_ann.net.history["loss_train"]);
    plt.xlabel("Iteration");
    plt.ylabel("$1-NSE$");
    plt.grid(alpha=.7, ls="--");
    @savefig user_guide.classical_uses.regionalization_spatial_validation.ann_J.png
    plt.title("Cost function descent");

Finally, we can visualize parameters and performances

.. ipython:: python

    f, ax = plt.subplots(2, 2)

    map_cp = ax[0,0].imshow(model_ann.get_rr_parameters("cp"));
    f.colorbar(map_cp, ax=ax[0,0], label="cp (mm)");
    map_ct = ax[0,1].imshow(model_ann.get_rr_parameters("ct"));
    f.colorbar(map_ct, ax=ax[0,1], label="ct (mm)");
    map_kexc = ax[1,0].imshow(model_ann.get_rr_parameters("kexc"));
    f.colorbar(map_kexc, ax=ax[1,0], label="kexc (mm/d)");
    map_llr = ax[1,1].imshow(model_ann.get_rr_parameters("llr"));
    @savefig user_guide.classical_uses.regionalization_spatial_validation.ann_theta.png
    f.colorbar(map_llr, ax=ax[1,1], label="llr (min)");

.. ipython:: python
    
    metrics = ["nse", "kge"]

    scores = np.round(smash.evaluation(model_ann, metrics)[1:, :], 2)

    upstream_perf = pd.DataFrame(data=scores, index=model.mesh.code[1:], columns=metrics)
    upstream_perf

.. ipython:: python
    :suppress:

    plt.close('all')
