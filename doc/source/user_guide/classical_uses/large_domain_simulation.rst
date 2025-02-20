.. _user_guide.quickstart.large_domain_simulation:

=======================
Large Domain Simulation
=======================

This tutorial aims to perform a simulation over the whole of metropolitan France with a simple model structure.
The objective is to create a mesh over a large spatial domain, to perform a forward run and to visualize the simulated discharge over the entire domain.
We begin by opening a Python interface:

.. code-block:: none

    python3

.. ipython:: python
    :suppress:

    import os

Imports
-------

We will first import the necessary libraries for this tutorial. Both ``LogNorm`` and ``SymLogNorm`` will be used for plotting.

.. ipython:: python

    import smash
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, SymLogNorm

Model creation
--------------

Now, we need to create a :class:`smash.Model` object.
For this case, we will use the :ref:`user_guide.data_and_format_description.france` dataset as an example.

Load the ``setup`` and ``mesh`` dictionaries using the `smash.factory.load_dataset` function and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.factory.load_dataset("France")
    model = smash.Model(setup, mesh)

Model simulation with a forward run
-----------------------------------

We can now call the `Model.forward_run <smash.Model.forward_run>` method, but by default and for memory reasons, the simulated discharge on the 
entire spatio-temporal domain is not saved. This means storing an `numpy.ndarray` of shape *(nrow, ncol, ntime_step)*, which may be quite large depending on the 
simulation period and the spatial domain. To activate this option, the ``return_options`` argument must be filled in, specifying that you want to retrieve 
the simulated discharge on the whole domain. Whenever the ``return_options`` is filled in, the `Model.forward_run <smash.Model.forward_run>` method
returns a `smash.ForwardRun` object storing these variables.

.. To speed up documentation generation
.. ipython:: python
    :suppress:
    
    ncpu = min(5, max(1, os.cpu_count() - 1))
    fwd_run = model.forward_run(return_options={"q_domain": True}, common_options={"ncpu": ncpu})

.. ipython:: python
    :verbatim:

    fwd_run = model.forward_run(return_options={"q_domain": True})

.. ipython:: python

    fwd_run
    fwd_run.time_step
    fwd_run.q_domain.shape

The returned object `smash.ForwardRun` contains two variables ``q_domain`` and ``time_step``. With ``q_domain`` a `numpy.ndarray` of shape 
*(nrow, ncol, ntime_step)* storing the simulated discharge and ``time_step`` a `pandas.DatetimeIndex` storing the saved time steps.
We can view the simulated discharge for one time step, for example the last one.

.. ipython:: python

    q = fwd_run.q_domain[..., -1]
    q = np.where(model.mesh.active_cell == 0, np.nan, q) # Remove the non-active cells from the plot
    plt.imshow(q, norm=SymLogNorm(1e-4));
    plt.colorbar(label="Discharge $(m^3/s)$");
    @savefig user_guide.quickstart.large_domain_simulation.forward_run_q.png
    plt.title("France - Discharge");

.. note::

    Given that we performed a forward run on only 32 time steps with default rainfall-runoff parameters and initial states, the simulated 
    discharge is not realistic.

By default, if the returned time steps are not defined, all the time steps are returned. It is possible to return only certain time steps by
specifying them in the ``return_options`` argument, for example only the two last ones.

.. ipython:: python

    time_step = ["2012-01-02 07:00", "2012-01-02 08:00"]  # define returned time steps

.. To speed up documentation generation
.. ipython:: python
    :suppress:
    
    ncpu = min(5, max(1, os.cpu_count() - 1))
    fwd_run = model.forward_run(return_options={"time_step": time_step, "q_domain": True}, common_options={"ncpu": ncpu})

.. ipython:: python
    :verbatim:

    fwd_run = model.forward_run(
        return_options={
            "time_step": time_step,
            "q_domain": True
        }
    )  # forward run and return q_domain at specified time steps

.. ipython:: python

    fwd_run.time_step
    fwd_run.q_domain.shape

.. ipython:: python

    q = fwd_run.q_domain[..., -1]
    q = np.where(model.mesh.active_cell == 0, np.nan, q) # Remove the non-active cells from the plot
    plt.imshow(q, norm=SymLogNorm(1e-4));
    plt.colorbar(label="Discharge $(m^3/s)$");
    @savefig user_guide.quickstart.large_domain_simulation.forward_run_q2.png
    plt.title("France - Discharge");

.. ipython:: python
    :suppress:

    plt.close('all')
