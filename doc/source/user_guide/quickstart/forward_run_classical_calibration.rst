.. _user_guide.in_depth.forward_run_classical_calibration:

=====================================
Forward Run and Classical Calibration
=====================================

.. warning::
    This section is in development.
    
.. TODO: PJ reprendre, improve ....

This tutorial explains how to perform simple simulations (forward run and simple optimization) and how to use several Input/Output operations. 
We begin by opening a Python interface:

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

Now, we need to create a :class:`smash.Model` object. 
For this case, we will use the :ref:`user_guide.data_and_format_description.cance` dataset as an example.

Load the ``setup`` and ``mesh`` dictionaries using the `smash.factory.load_dataset` function and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.factory.load_dataset("Cance")
    model = smash.Model(setup, mesh)

Model simulation
----------------

Different methods associated with the `smash.Model` object are available to perform a simulation such as a forward run or an optimization.

Forward run
***********

The most basic simulation possible is the forward run that consists of running a forward hydrological model given input data. A forward run can be called with the `Model.forward_run <smash.Model.forward_run>` method.

.. To speed up documentation generation
.. ipython:: python
    :suppress:

    ncpu = min(5, max(1, os.cpu_count() - 1))
    model.forward_run(common_options={"ncpu": ncpu})

.. ipython:: python
    :verbatim:

    model.forward_run()

Once the forward run has been completed, we can visualize the simulated discharge, for example, at the most downstream gauge.

.. ipython:: python

    code = model.mesh.code[0]
    plt.plot(model.response_data.q[0, :], label="Observed discharge");
    plt.plot(model.response.q[0, :], label="Simulated discharge");
    plt.xlabel("Time step");
    plt.ylabel("Discharge ($m^3/s$)");
    plt.grid(ls="--", alpha=.7);
    plt.legend();
    @savefig user_guide.in_depth.classical_calibration_io.forward_run_q.png
    plt.title(f"Observed and simulated discharge at gauge {code}");

As the hydrograph shows, the simulated discharge is quite different from the observed discharge at this gauge. Obviously, we ran a forward run with the default `smash` rainfall-runoff 
parameter set. We can now try to run an optimization to minimize the misfit between the simulated and observed discharge. 

Optimization
************

Similar to the `Model.forward_run <smash.Model.forward_run>` method, an optimization can be called with the `Model.optimize <smash.Model.optimize>` method.

.. To speed up documentation generation
.. ipython:: python
    :suppress:

    ncpu = min(5, max(1, os.cpu_count() - 1))
    model.optimize(common_options={"ncpu": ncpu})

.. ipython:: python
    :verbatim:

    model.optimize()

And visualize again the simulated discharge compared to the observed discharge, but this time with optimized model parameters.

.. ipython:: python

    code = model.mesh.code[0]
    plt.plot(model.response_data.q[0, :], label="Observed discharge");
    plt.plot(model.response.q[0, :], label="Simulated discharge");
    plt.xlabel("Time step");
    plt.ylabel("Discharge ($m^3/s$)");
    plt.grid(ls="--", alpha=.7);
    plt.legend();
    @savefig user_guide.in_depth.classical_calibration_io.optimize_q.png
    plt.title(f"Observed and simulated discharge at gauge {code}");

Of course, the hydrological model optimization problem is a complex one and there are many strategies that can be employed depending on the modeling goals and data available. Here, for a first tutorial, we have run a simple optimization with the function's
default parameters (``SBS`` global :ref:`optimization algorithm <math_num_documentation.optimization_algorithm>`). The end of this section will be dedicated to a brief explanation of the information associated with the optimization performed.

First, several pieces of information were displayed on the screen during optimization:

.. code-block:: text

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

These lines show the different iterations of the optimization with information on the number of iterations, the number of cumulative evaluations ``nfg`` 
(number of forward runs performed within each iteration of the optimization algorithm), the value of the cost function to minimize ``J`` and the value of the adaptive descent step ``ddx`` of this heuristic search algorithm. 
So, to summarize, the optimization algorithm has converged after 8 iterations by reaching the descent step tolerance criterion of 0.01. This optimization required performing 240 forward run evaluations and leads to a final cost function value of 0.0367.

Then, we can ask which cost function ``J`` has been minimized and which parameters have been optimized. So, by default, the cost function to be minimized is one minus the Nash-Sutcliffe efficiency ``nse`` (:math:`1 - \text{NSE}`)
and the optimized parameters are the set of rainfall-runoff parameters (``cp``, ``ct``, ``kexc`` and ``llr``). In the current configuration, spatially
uniform parameters were optimized, i.e., a spatially uniform map for each parameter. We can visualize the optimized rainfall-runoff parameters.

.. ipython:: python

    ind = tuple(model.mesh.gauge_pos[0, :])
    opt_parameters = {
        k: model.get_rr_parameters(k)[ind] for k in ["cp", "ct", "kexc", "llr"]
    } # A dictionary comprehension
    opt_parameters

Save Model
----------

Finally, like the ``setup`` and ``mesh`` dictionaries, the `smash.Model` object, including the optimized parameters, can be saved to `HDF5 <https://www.hdfgroup.org/solutions/hdf5>`__ format
and read back using the `smash.io.save_model` and `smash.io.read_model` functions, respectively.

.. ipython:: python

    smash.io.save_model(model, "model.hdf5")
    model = smash.io.read_model("model.hdf5")
    model

.. ipython:: python
    :suppress:

    plt.close('all')
