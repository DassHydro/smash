.. _user_guide.in_depth.optimize.bayes_estimate:

===================================================
Improving the first guess using Bayesian estimation
===================================================

Here, we aim to improve the spatially uniform first guess using the Bayesian estimation approach.

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

.. _user_guide.indepth.optimize.ld-estim:

------------------------------------------
LDBE (Low Dimensional Bayesian Estimation)
------------------------------------------

Next, we find a uniform first guess using Bayesian estimation on a random set of the Model parameters using the :meth:`smash.Model.bayes_estimate` method. 
By default, the ``sample`` argument is not required to be set, and it is automatically set based on the Model structure. 
This example shows how to define a custom ``sample`` using the :meth:`smash.generate_samples` method. 

You may refer to the :meth:`smash.Model.get_bound_constraints` method to obtain some information about the Model parameters/states:

.. ipython:: python

    model.get_bound_constraints()

Here we define a problem that only contains three Model parameters, which means the fourth one will be fixed:

.. ipython:: python

    problem = {
            "num_vars": 3,
            "names": ["cp", "lr", "cft"],
            "bounds": [[1, 1000], [1, 1000], [1, 1000]]
        }

We then generate a set of 400 random Model parameters:

.. ipython:: python

    sr = smash.generate_samples(problem, n=400, random_state=1)

and perform Bayesian estimation:

.. ipython:: python

    model_be, br = model.bayes_estimate(sr, alpha=np.linspace(-1, 4, 50), return_br=True);
    model_be.output.cost  # cost value with LDBE

In the code above, we used the L-curve approach to find an optimal regularization parameter within a short search range of :math:`[-1, 4]`.

-----------------------------------
Visualization of estimation results
-----------------------------------

Now, we can use the ``br`` instance of :class:`smash.BayesResult` to visualize information about the estimation process. 
For example, we can plot the distribution of cost values obtained from running the forward hydrological model 
with the random set of parameters using the following code: 

.. ipython:: python

    plt.hist(br.data["cost"], bins=30, zorder=2);
    plt.grid(alpha=.7, ls="--", zorder=1);
    plt.xlabel("Cost");
    plt.ylabel("Frequency");
    @savefig user_guide.in_depth.optimize.bayes_estimate.cost_distribution.png
    plt.title("Cost value histogram for parameter set");

The minimal cost value through the forward runs:

.. ipython:: python

    np.min(br.data["cost"])

We can also visualize the L-curve that was used to find the optimal regularization parameter:

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
    @savefig user_guide.in_depth.optimize.bayes_estimate.lcurve.png
    plt.legend();

The spatially uniform first guess:

.. ipython:: python

    ind = tuple(model_be.mesh.gauge_pos[0,:])
    
    ind

    (
     model_be.parameters.cp[ind],
     model_be.parameters.cft[ind],
     model_be.parameters.exc[ind],
     model_be.parameters.lr[ind],
    )

Comparing to the initial values of the parameters:

.. ipython:: python

    (
     model.parameters.cp[ind],
     model.parameters.cft[ind],
     model.parameters.exc[ind],
     model.parameters.lr[ind],
    )

We can see that the value of ``exc`` did not change since it is not set to be estimated.

.. ipython:: python
    :suppress:

    if not np.allclose(model.parameters.exc, model_be.parameters.exc, atol=1e-08):
        raise AssertionError("Non-estimated parameter changes its values in bayes_estimate!")
    if not np.allclose(
            (
                model_be.parameters.cp[ind],
                model_be.parameters.cft[ind],
                model_be.parameters.exc[ind],
                model_be.parameters.lr[ind],
            ),
            (112.33628, 99.58623, 0.0, 518.99603),
            atol=1e-04
        ):  # This check is used to verify the value of unif_backg in bayes_optimize
        raise AssertionError("Estimated parameters have been changed in bayes_estimate!")

-------------------------------------------------------------
Variational calibration using Bayesian estimation first guess
-------------------------------------------------------------

Finally, we perform a variational calibration on the Model parameters using 
the :math:`\mathrm{L}\text{-}\mathrm{BFGS}\text{-}\mathrm{B}` algorithm and the Bayesian first guess:

.. ipython:: python
    :suppress:

    model_sd = model_be.optimize(
            mapping="distributed", 
            algorithm="l-bfgs-b", 
            options={"maxiter": 30}
        )

.. ipython:: python
    :verbatim:

    model_sd = model_be.optimize(
        mapping="distributed", 
        algorithm="l-bfgs-b", 
        options={"maxiter": 30}
    )

.. ipython:: python

    model_sd.output.cost  # the cost value

The spatially distributed model parameters:

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
    @savefig user_guide.in_depth.optimize.bayes_estimate.theta.png
    f.colorbar(map_exc, ax=ax[1,1], label="exc (mm/d)");

.. ipython:: python
    :suppress:

    plt.close('all')
