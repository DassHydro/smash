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
            k=np.linspace(-1, 8, 20), 
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

    plt.hist(br.data["cost"], range=[0, 2]);  # limit cost range 
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Cost");
    plt.ylabel("Frequency");
    @savefig distribution_cost_be_user_guide.png
    plt.title("Distribution of cost values on the parameters set");

We can also visualize the L-curve that was used to find the optimal regularization parameter:

.. ipython:: python

    plt.scatter(
            br.l_curve["k"], 
            br.l_curve["cost"], 
            label="Regularization parameter"
        );
    plt.scatter(
            br.l_curve["k_opt"], 
            model_be.output.cost, 
            color="red", 
            label="Optimal regularization parameter"
        );
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("k");
    plt.ylabel("Cost");
    plt.title("L-curve");
    @savefig lcurve_estimate_be_user_guide.png
    plt.legend();

-------------------------------------------------------------
Variational calibration using Bayesian estimation first guess
-------------------------------------------------------------

Finally, we perform a variational calibration on the model parameters using the ``L-BFGS-B`` algorithm with 
the Bayesian first guess:

.. ipython:: python
    :suppress:

    model_sd = model_be.optimize(
            mapping="distributed", 
            algorithm="l-bfgs-b", 
            options={"maxiter": 30}
        )

.. ipython:: python
    :verbatim:

    model_hp = model_su.optimize(
            mapping="hyper-polynomial", 
            algorithm="l-bfgs-b", 
            options={"maxiter": 30}
        )

.. ipython:: python

    model_sd.output.cost  # the cost value
