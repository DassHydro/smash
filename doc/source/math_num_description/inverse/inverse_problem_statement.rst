.. _math_num_description.inverse_problem_statement :

=========================
Inverse problem statement
=========================

.. warning::
   To be extended/generalized vs proba cost func, bayesian, mcmc, ...

Hydrological optimization problems generally consist in fitting catchment model response to the available observations, that is potentially multi-source and multi-site heterogeneous data used to inform the model. The misfit between observations and model response is measured with a cost function :math:`J` (cf. :ref:`Cost functions <math_num_description.cost_functions>`) that depends on hydrological parameters :math:`\theta` through the forward hydrological model :math:`\mathcal{M}` composed of hydrological operators that can include a pre-regionalization (cf. :ref:`Forward problem statement <math_num_description.forward_problem_statement>`). The inverse problem consists in searching an optimal parameter set :math:`\hat{\boldsymbol{\theta}}` such that:

.. math::
   :name: eq:3
   
   \hat{\boldsymbol{\theta}}=\arg\min_{\boldsymbol{\theta}}J\left(\boldsymbol{\theta}\right).

**TODO**. add control vectors hyperparam ... 
