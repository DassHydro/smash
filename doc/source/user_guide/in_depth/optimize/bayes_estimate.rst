.. _user_guide.optimize.bayes_estimate:

===================================================
Improving the first guess using Bayesian estimation
===================================================

.. .. ipython:: python

..     model_ufg = model.bayes_estimate(k=np.linspace(-2, 12, 50), n=200, random_state=0)

.. In the example above, we generated a set of 200 random spatially uniform model parameters, and 
.. used the L-curve approach to find an optimal regularization parameter within a search range of :math:`[-2, 12]`.