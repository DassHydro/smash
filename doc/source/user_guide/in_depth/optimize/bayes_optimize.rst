.. _user_guide.in_depth.optimize.bayes_optimize:

================================
Variational Bayesian calibration
================================

Here, we aim to optimize the Model parameters/states using a variational Bayesian calibration approach.

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

First, you need to create a :class:`smash.Model` object. 
For this case, we will use the ``Lez`` dataset.

Load the ``setup`` and ``mesh`` dictionaries using the :meth:`smash.load_dataset` method and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.load_dataset("Lez")
    
    model = smash.Model(setup, mesh)

------------------
Generating samples
------------------

By default, the ``sample`` argument is not required to be set, and it is automatically set based on both ``control_vector`` and ``bounds`` arguments in the :class:`smash.Model.bayes_optimize` method. 
This example shows how to define a custom ``sample`` using the :meth:`smash.generate_samples` method. 

The problem to generate samples must be defined first. The default Model parameters and bounds condition can be obtained using 
the :meth:`smash.Model.get_bound_constraints` method as follows:

.. ipython:: python

    problem = model.get_bound_constraints()
    problem

Then, we generate samples using the :meth:`smash.generate_samples` method. For instance, we can use the Gaussian distribution with means defined by a prior uniform background solution.
This uniform background can be obtained through a global optimization algorithm or an :ref:`LD-Bayesian Estimation <indepth.optimize.ld-estim>` approach. 
In this example, we use a prior solution obtained via the LDBE approach:

.. ipython:: python

    unif_backg = dict(
            zip(
                ("cp", "cft", "exc", "lr"), 
                (112.33628, 99.58623, 0.0, 518.99603)
            )
        )

    unif_backg

Next, we generate samples using the :meth:`smash.generate_samples` method:

.. ipython:: python

    sr = smash.generate_samples(
            problem, 
            generator="normal", 
            n=100, 
            mean=unif_backg,
            coef_std=12, 
            random_state=23
        )

.. note::

    In real-world applications, the number of generated samples ``n`` can be much larger to attain more accurate results when applying the ``sr`` object to the :class:`smash.Model.bayes_optimize` method.

Note that if the mean value is too close to the bounds of the distribution, we use a truncated Gaussian distribution to generate samples, ensuring that they do not exceed their bounds. 
The distribution of generated samples can be visualized as shown below:

.. ipython:: python

    f, ax = plt.subplots(2, 2, figsize=(6.4, 6.4))

    ax[0, 0].hist(sr.cp, bins=30);
    ax[0, 0].set_xlabel("cp (mm)");
    ax[0, 0].set_ylabel("Frequency");

    ax[0, 1].hist(sr.cft, bins=30);
    ax[0, 1].set_xlabel("cft (mm)");

    ax[1, 0].hist(sr.lr, bins=30, label="lr");
    ax[1, 0].set_xlabel("lr (min)");
    ax[1, 0].set_ylabel("Frequency");

    ax[1, 1].hist(sr.exc, bins=30, label="lr");
    @savefig user_guide.in_depth.optimize.bayes_optimize.gen_param_distribution.png
    ax[1, 1].set_xlabel("exc (mm/d)");

.. ipython:: python
    :suppress:

    plt.figure(figsize=plt.rcParamsDefault['figure.figsize'])  # Reset figsize to the Matplotlib default

--------------------------------------------
HDBC (High Dimensional Bayesian Calibration)
--------------------------------------------

Once the samples are created in the ``sr`` object, we can employ an HDBC approach that incoporates multiple calibrations with VDA (using the :math:`\mathrm{L}\text{-}\mathrm{BFGS}\text{-}\mathrm{B}` algorithm) and Bayesian estimation in high dimension. 
It can be implemented using the :class:`smash.Model.bayes_optimize` method as follows:

.. ipython:: python

    model_bo, br = model.bayes_optimize(
            sr,
            alpha=np.linspace(-1, 16, 60),
            mapping="distributed",
            algorithm="l-bfgs-b",
            options={"maxiter": 4},
            return_br=True
        );

    model_bo.output.cost  # cost value with HDBC

.. note::

    In real-world applications, the maximum allowed number of iterations ``options["maxiter"]`` 
    can be much larger to attain more accurate results.

------------------------
Visualization of results
------------------------

To visualize information about the Bayesian estimation process, we can use the ``br`` instance of :class:`smash.BayesResult`. 
For instance, to display the histogram of the cost values when calibrating the Model parameters using the generated samples:

.. ipython:: python

    plt.hist(br.data["cost"], bins=30, zorder=2);
    plt.grid(alpha=.7, ls="--", zorder=1);
    plt.xlabel("Cost");
    plt.ylabel("Frequency");
    @savefig user_guide.in_depth.optimize.bayes_optimize.hist_cost.png
    plt.title("Cost value histogram for parameter set");

The minimal cost value through multiple calibrations:

.. ipython:: python

    np.min(br.data["cost"])

Then, can also visualize the L-curve that was used to find the optimal regularization parameter:

.. ipython:: python

    opt_ind = np.where(br.lcurve["alpha"]==br.lcurve["alpha_opt"])[0][0]
    plt.scatter(
            br.lcurve["mahal_dist"], 
            br.lcurve["cost"],
            label="Regularization parameter",
            zorder=2
        );
    plt.scatter(
            br.lcurve["mahal_dist"][opt_ind], 
            br.lcurve["cost"][opt_ind], 
            color="red", 
            label="Optimal value",
            zorder=3
        );
    plt.grid(alpha=.7, ls="--", zorder=1);
    plt.xlabel("Mahalanobis distance");
    plt.ylabel("Cost");
    plt.title("L-curve");
    @savefig user_guide.in_depth.optimize.bayes_optimize.lcurve.png
    plt.legend();

Finally, the spatially distributed model parameters can be visualized using the ``model_bo`` object:

.. ipython:: python

    ma = (model_bo.mesh.active_cell == 0)

    ma_cp = np.where(ma, np.nan, model_bo.parameters.cp)
    ma_cft = np.where(ma, np.nan, model_bo.parameters.cft)
    ma_lr = np.where(ma, np.nan, model_bo.parameters.lr)
    ma_exc = np.where(ma, np.nan, model_bo.parameters.exc)

    f, ax = plt.subplots(2, 2)

    map_cp = ax[0,0].imshow(ma_cp);
    f.colorbar(map_cp, ax=ax[0,0], label="cp (mm)");

    map_cft = ax[0,1].imshow(ma_cft);
    f.colorbar(map_cft, ax=ax[0,1], label="cft (mm)");

    map_lr = ax[1,0].imshow(ma_lr);
    f.colorbar(map_lr, ax=ax[1,0], label="lr (min)");

    map_exc = ax[1,1].imshow(ma_exc);
    @savefig user_guide.in_depth.optimize.bayes_optimize.theta.png
    f.colorbar(map_exc, ax=ax[1,1], label="exc (mm/d)");

.. ipython:: python
    :suppress:

    plt.close('all')