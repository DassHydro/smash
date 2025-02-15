.. _user_guide.classical_uses.split_sample_temporal_validation:

====================================
Split Sample and Temporal Validation
====================================

The objective of this tutorial is to learn how to set up an optimization with a split sample test :cite:p:`klemevs1986operational` in `smash`,
i.e., cross-calibration and temporal validation over two distinct periods ``p1`` and ``p2``.

Open a Python interface:

.. code-block:: none

    python3

.. ipython:: python
    :suppress:

    import os

Imports
-------

We will first import the necessary libraries for this tutorial.

.. ipython:: python

    import smash
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

Model creation
--------------

In this tutorial, we will use the :ref:`user_guide.data_and_format_description.lez` dataset as an example.

Model creation
**************

Since we are going to work on two different periods, each of 6 months, we need to create two ``setup`` dictionaries where the only difference 
will be in the simulation period arguments ``start_time`` and ``end_time``. The first period ``p1`` will run from ``2012-08-01`` to
``2013-01-31`` and the second, from ``2013-02-01`` to ``2013-07-31``.

.. ipython:: python

    setup, mesh = smash.factory.load_dataset("lez")  # load setup and mesh
    setup["start_time"], setup["end_time"]

    setup_p1 = setup.copy()
    setup_p1.update(
        {
            "start_time": "2012-08-01",
            "end_time": "2013-01-31",
        }
    )

    setup_p2 = setup.copy()
    setup_p2.update(
        {
            "start_time": "2013-02-01",
            "end_time": "2013-07-31",
        }
    )

For the ``mesh``, there is no need to generate two different ``mesh`` dictionaries, the same one can be used for both time periods.
Now we can initialize the two `smash.Model` objects:

.. ipython:: python

    model_p1 = smash.Model(setup_p1, mesh)
    model_p2 = smash.Model(setup_p2, mesh)

Model simulation
----------------

Optimization
************

First, we will optimize both models for each period to generate two sets of optimized rainfall-runoff parameters.
So far, to optimize, we have called the method associated with the `smash.Model` object `Model.optimize <smash.Model.optimize>`. This method
will modify the associated object in place (i.e., the values of the rainfall-runoff parameters after calling this function are modified). Here, we
want to optimize the model but still keep this model object to run the validation afterwards. To do this, instead of calling the
`Model.optimize <smash.Model.optimize>` method, we can call the `smash.optimize` function, which is identical but takes a
`smash.Model` object as input and returns a copy of it. This method allows you to optimize a `smash.Model` object and store the results in 
another object without modifying the initial one.

Here, we will perform a simple spatially uniform optimization (``SBS`` global :ref:`optimization algorithm <math_num_documentation.optimization_algorithm>`) of the rainfall-runoff parameters
by minimizing the cost function equal to one minus the Nash-Sutcliffe efficiency on the most downstream gauge.

.. To speed up documentation generation
.. ipython:: python
    :suppress:

    ncpu = min(5, max(1, os.cpu_count() - 1))
    model_p1_opt = smash.optimize(model_p1, common_options={"ncpu": ncpu})
    model_p2_opt = smash.optimize(model_p2, common_options={"ncpu": ncpu})

.. ipython:: python
    :verbatim:

    model_p1_opt = smash.optimize(model_p1)
    model_p2_opt = smash.optimize(model_p2)

We can take a look at the hydrographs and optimized rainfall-runoff parameters.

.. ipython:: python

    code = model_p1.mesh.code[0]

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4));

    qobs = model_p1_opt.response_data.q[0,:].copy()
    qobs = np.where(qobs < 0, np.nan, qobs) # To deal with missing values
    qsim = model_p1_opt.response.q[0,:]
    ax1.plot(qobs);
    ax1.plot(qsim);
    ax1.grid(ls="--", alpha=.7);
    ax1.set_xlabel("Time step");
    ax1.set_ylabel("Discharge ($m^3/s$)");

    qobs = model_p2_opt.response_data.q[0,:].copy()
    qobs = np.where(qobs < 0, np.nan, qobs) # To deal with missing values
    qsim = model_p2_opt.response.q[0,:]
    ax2.plot(qobs, label="Observed discharge");
    ax2.plot(qsim, label="Simulated discharge");
    ax2.grid(ls="--", alpha=.7);
    ax2.set_xlabel("Time step");
    ax2.legend();

    @savefig user_guide.classical_uses.split_sample_temporal_validation.optimize_q.png
    f.suptitle(
        f"Observed and simulated discharge at gauge {code}"
        " for period p1 (left) and p2 (right)\nCalibration"
    );

.. ipython:: python

    ind = tuple(model_p1.mesh.gauge_pos[0, :])

    opt_parameters_p1 = {
        k: model_p1_opt.get_rr_parameters(k)[ind] for k in ["cp", "ct", "kexc", "llr"]
    } # A dictionary comprehension

    opt_parameters_p2 = {
        k: model_p2_opt.get_rr_parameters(k)[ind] for k in ["cp", "ct", "kexc", "llr"]
    } # A dictionary comprehension

    opt_parameters_p1
    opt_parameters_p2

Temporal validation
*******************

Rainfall-runoff parameters transfer
'''''''''''''''''''''''''''''''''''

We can now transfer the optimized rainfall-runoff parameters for each calibration period to the respective validation period. 
We will transfer the rainfall-runoff parameters from ``model_p1_opt`` to ``model_p2`` and from ``model_p2_opt`` to ``model_p1``. 
There are several ways to do this:

- Transfer all rainfall-runoff parameters at once
    All rainfall-runoff parameters are stored in the variable ``values`` of the object `Model.rr_parameters <smash.Model.rr_parameters>`. 
    We can therefore pass the whole array of rainfall-runoff parameters from one object to the other.

    .. ipython:: python

        model_p1.rr_parameters.values = model_p2_opt.rr_parameters.values.copy()
        model_p2.rr_parameters.values = model_p1_opt.rr_parameters.values.copy()

    .. note::
        A deep copy is recommended to avoid that the rainfall-runoff parameters between each object become shallow copies and
        so that the modification of one of the arrays leads to the modification of another.

- Transfer each rainfall-runoff parameter one by one
    It is also possible to loop on each rainfall-runoff parameter and assign new rainfall-runoff parameter by passing
    by getters and setters

    .. ipython:: python

        for key in model_p1.rr_parameters.keys:
            model_p1.set_rr_parameters(key, model_p2_opt.get_rr_parameters(key))
            model_p2.set_rr_parameters(key, model_p1_opt.get_rr_parameters(key))

    .. note::
        This method allows, instead of looping on all rainfall-runoff parameters, to loop only on some. We can replace
        ``model_p1.rr_parameters.keys`` by ``["cp", "ct"]`` for example

Forward run
'''''''''''

Once the rainfall-runoff parameters have been transferred, we can proceed with the validation forward runs by calling the 
`Model.forward_run <smash.Model.forward_run>` method.

.. ipython:: python

    model_p1.forward_run()
    model_p2.forward_run()

and visualize hydrographs

.. ipython:: python

    code = model_p1.mesh.code[0]

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4));

    qobs = model_p1.response_data.q[0,:].copy()
    qobs = np.where(qobs < 0, np.nan, qobs) # To deal with missing values
    qsim = model_p1.response.q[0,:]
    ax1.plot(qobs);
    ax1.plot(qsim);
    ax1.grid(ls="--", alpha=.7);
    ax1.set_xlabel("Time step");
    ax1.set_ylabel("Discharge ($m^3/s$)");

    qobs = model_p2.response_data.q[0,:].copy()
    qobs = np.where(qobs < 0, np.nan, qobs) # To deal with missing values
    qsim = model_p2.response.q[0,:]
    ax2.plot(qobs, label="Observed discharge");
    ax2.plot(qsim, label="Simulated discharge");
    ax2.grid(ls="--", alpha=.7);
    ax2.set_xlabel("Time step");
    ax2.legend();

    @savefig user_guide.classical_uses.split_sample_temporal_validation.forward_run_q.png
    f.suptitle(
        f"Observed and simulated discharge at gauge {code}"
        " for period p1 (left) and p2 (right)\nValidation"
    );

Model performances
------------------

We evaluate calibration and validation performances using certain metrics. Using the function `smash.evaluation`,
you can compute one metric of your choice (among those available) for all the gauges that make up the ``mesh``. Here, we are interested 
in the ``nse`` (the calibration metric) and the ``kge`` for the downstream gauge only. We will create two `pandas.DataFrame`, one for the 
calibration performances and the other for the validation performances.

.. ipython:: python

    metrics = ["nse", "kge"]
    perf_cal = pd.DataFrame(index=["p1", "p2"], columns=metrics)
    perf_val = perf_cal.copy()

    perf_cal.loc["p1"] = np.round(smash.evaluation(model_p1_opt, metrics)[0, :], 2)
    perf_cal.loc["p2"] = np.round(smash.evaluation(model_p2_opt, metrics)[0, :], 2)

    perf_val.loc["p1"] = np.round(smash.evaluation(model_p1, metrics)[0, :], 2)
    perf_val.loc["p2"] = np.round(smash.evaluation(model_p2, metrics)[0, :], 2)

    perf_cal # Calibration performances

    perf_val # Validation performances

.. TODO: Add a conclusion (or change case ...) on this split sample test parameters are wildly different... I suspect it's due to
.. the state initialisation (Qobs is quite high at the beginning of p2). Not a big deal in the context of this doc,
.. but it could be mentioned either here or maybe better as a conclusion of this split-sample exercise, to demonstrate.
.. its utility and explain why the validation metrics are quite bad.

.. ipython:: python
    :suppress:

    plt.close('all')