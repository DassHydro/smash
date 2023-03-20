.. _user_guide.optimize.bayes_estimate:

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

-------------------
Bayesian estimation
-------------------

Next, we find a uniform first guess using Bayesian estimation on a random set of the model parameters:

.. ipython:: python

    model_be, br = model.bayes_estimate(
            k=np.linspace(-1, 8, 50), 
            n=400, 
            random_state=11, 
            return_br=True
        )

In the code above, we generated a set of 400 random model parameters, and 
used the L-curve approach to find an optimal regularization parameter within a short search range of :math:`[-1, 8]`.

-----------------------------------
Visualization of estimation results
-----------------------------------

Now, we can use the ``br`` instance of :class:`smash.BayesResult` to visualize information about the estimation process. 
For example, we can plot the distribution of cost values obtained from running the forward hydrological model 
with the random set of parameters using the following code: 

.. ipython:: python

    plt.hist(br.data["cost"], range=[0, 2], zorder=2);  # limit cost range 
    plt.grid(alpha=.7, ls="--", zorder=1);
    plt.xlabel("Cost");
    plt.ylabel("Frequency");
    @savefig distribution_cost_be_user_guide.png
    plt.title("Distribution of cost values on the parameters set");

We can also visualize the L-curve that was used to find the optimal regularization parameter:

.. ipython:: python

    opt_ind = np.where(br.l_curve["k"]==br.l_curve["k_opt"])[0][0]
    plt.scatter(br.l_curve["Mahalanobis_distance"], br.l_curve["cost"]);
    plt.scatter(
            br.l_curve["Mahalanobis_distance"][opt_ind], 
            br.l_curve["cost"][opt_ind], 
            color="red", 
            label="Optimal regularization point"
        );
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Mahalanobis distance");
    plt.ylabel("Cost");
    plt.title("L-curve");
    @savefig lcurve_estimate_be_user_guide.png
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

-------------------------------------------------------------
Variational calibration using Bayesian estimation first guess
-------------------------------------------------------------

Finally, we perform a variational calibration on the model parameters using 
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
    @savefig user_guide.in_depth.optimize.bayes_estimate.theta_sd.png
    f.colorbar(map_exc, ax=ax[1,1], label="exc (mm/d)");
