.. _user_guide.optimize.fully_distributed:

==========================================================
Fully-distributed optimization using a uniform first guess
==========================================================

To get started, open a Python interface:

.. code-block:: none

    python3
    
-------
Imports
-------

.. ipython:: python
    
    import smash
    import matplotlib.pyplot as plt
    import numpy as np

---------------------
Model object creation
---------------------

To perform the calibrations, you need to create a :class:`smash.Model` object. 
For this case, we will use the ``Lez`` dataset.

Load the ``setup`` and ``mesh`` dictionaries using the :meth:`smash.load_dataset` method and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.load_dataset("Lez")
    
    model = smash.Model(setup, mesh)

------------------------------
Spatially uniform optimization
------------------------------

To find the uniform first guess, you can perform a spatially-uniform calibration using global optimization algorithms. 
Here's an example how you can do it with the SBS algorithm:

.. ipython:: python

    model_su = model.optimize(mapping="uniform", algorithm="sbs", options={"maxiter": 2});

Once the optimization is complete. We can visualize the simulated discharge:

.. ipython:: python

    qo = model_su.input_data.qobs[0,:].copy()
    qo = np.where(qo<0, np.nan, qo)  # to deal with missing data
    plt.plot(qo, label="Observed discharge");
    plt.plot(model_su.output.qsim[0,:], label="Simulated discharge");
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    plt.title(model_su.mesh.code[0]);
    @savefig qsim_su_optimize_fd_user_guide.png
    plt.legend();
    
And the cost function value :math:`J`:

.. ipython:: python

    model_su.output.cost

----------------------------------
Spatially distributed optimization
----------------------------------

Next, using the first guess provided by a global calibration, which had stored the optimized parameters 
in the previous step, we perform a spatially distributed calibration using ``L-BFGS-B`` algorithm:

.. ipython:: python
    :suppress:

    model_sd = model_su.optimize(
            mapping="distributed", 
            algorithm="l-bfgs-b", 
            options={"maxiter": 30}
        )

.. ipython:: python
    :verbatim:

    model_sd = model_su.optimize(
            mapping="distributed", 
            algorithm="l-bfgs-b", 
            options={"maxiter": 30}
        )

We can once again visualize, the simulated discharges (``su``: spatially uniform, ``sd``: spatially distributed):

.. ipython:: python

    qo = model_sd.input_data.qobs[0,:].copy()
    qo = np.where(qo<0, np.nan, qo)  # to deal with missing data
    plt.plot(qo, label="Observed discharge");
    plt.plot(model_su.output.qsim[0,:], label="Simulated discharge - su");
    plt.plot(model_sd.output.qsim[0,:], label="Simulated discharge - sd");
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    plt.title(model_sd.mesh.code[0]);
    @savefig qsim_sd_optimize_fd_user_guide.png
    plt.legend();

The cost value:

.. ipython:: python

    model_sd.output.cost

And finally, the distributed model parameters in this case:

.. ipython:: python

    ma = (model_sd.mesh.active_cell == 0)

    ma_cp = np.where(ma, np.nan, model_sd.parameters.cp)
    ma_cft = np.where(ma, np.nan, model_sd.parameters.cft)
    ma_lr = np.where(ma, np.nan, model_sd.parameters.lr)
    ma_exc = np.where(ma, np.nan, model_sd.parameters.exc)
    
    f, ax = plt.subplots(2, 2)
    
    map_cp = ax[0,0].imshow(ma_cp);
    f.colorbar(map_cp, ax=ax[0,0], label="cp (mm)");
    
    map_cft = ax[0,1].imshow(ma_cft);
    f.colorbar(map_cft, ax=ax[0,1], label="cft (mm)");
    
    map_lr = ax[1,0].imshow(ma_lr);
    f.colorbar(map_lr, ax=ax[1,0], label="lr (min)");
    
    map_exc = ax[1,1].imshow(ma_exc);
    @savefig theta_sd_optimize_fd_user_guide.png
    f.colorbar(map_exc, ax=ax[1,1], label="exc (mm/h)");

.. note::

    Note that the variational calibration algorithm was unable to spatialize the linear routing parameter ``lr`` in this case. 
    This could be due to the fact that the uniform first guess obtained by the SBS algorithm may not be appropriate for certain situations. 
    Consider referring to the :ref:`Bayesian estimation <user_guide.optimize.bayes_estimate>` section 
    for information on how to improve the first guess using a Bayesian estimation approach.